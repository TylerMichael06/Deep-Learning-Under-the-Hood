"""Backend integration: training, masked inference, and CPU-owned datasets."""
import numpy as np
import pytest
import torch
import time
from argparse import Namespace
from obdfault import features, model
from obdfault.config import CH, W
from synthdata import make_trips, EVAL_ARGS

PHYSICS_INPUTS = features.PHYSICS[:]


def test_unavailable_gpus_are_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert model.resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        model.resolve_device("mps")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        model.resolve_device("cuda")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("physics", [False, True])
def test_training_and_inference(device, physics, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    inputs = PHYSICS_INPUTS[:] if physics else []
    monkeypatch.setattr(features, "PHYSICS", inputs)
    monkeypatch.setattr(model, "PHYSICS", inputs)
    torch.manual_seed(42)
    n, c = 32, len(CH) + len(inputs)
    D = dict(x=torch.randn(n, W, c), m=torch.ones(n, c), ctx=torch.zeros(n, W))
    D["m"][:4, 0] = 0
    net = model.VirtualSensorNet().to(device)
    model.fit(net, D, np.arange(24), np.arange(24, n), epochs=1, lr=1e-3, bs=16)
    errors = model.channel_errors(net, D, bs=16)
    assert np.isnan(errors[:4, 0]).all()
    assert np.isfinite(errors[4:]).all()
    assert all(t.device.type == "cpu" for t in D.values())
    cpu = model.VirtualSensorNet()
    cpu.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
    np.testing.assert_allclose(errors, model.channel_errors(cpu, D, bs=16), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_checkpoint_loads_on_cpu(device, tmp_path, monkeypatch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from obdfault import train
    monkeypatch.setattr(train, "MODELS", str(tmp_path))
    args = Namespace(**vars(EVAL_ARGS), device=device)
    train.save_final(make_trips(), args, time.time())
    checkpoint = torch.load(tmp_path / "obd_pretrained.pt", weights_only=False)
    assert all(t.device.type == "cpu" for t in checkpoint["state_dict"].values())
    net = model.VirtualSensorNet()
    net.load_state_dict(checkpoint["state_dict"])


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (True, False, "cuda"),
    (False, True, "mps"), (False, False, "cpu"),
])
def test_auto_backend_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert model.resolve_device("auto").type == expected
    assert model.resolve_device("cpu").type == "cpu"
