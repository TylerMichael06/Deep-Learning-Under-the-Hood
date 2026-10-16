"""Compare CPU/GPU on the same real-data pretrain, fine-tune, and clean inference.

After installing the package, run python scripts/benchmark_devices.py. No official outputs
are written. Fixed epoch counts and seeds; CPU/GPU random streams can differ.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from obdfault.config import KAGGLE_DATA
from obdfault.data import load_kaggle
from obdfault.features import PHYSICS
from obdfault.model import VirtualSensorNet, channel_errors, fit, resolve_device
from obdfault.train import prepare_car


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=KAGGLE_DATA)
    ap.add_argument("--car", default="car11")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--device", choices=["auto", "mps", "cuda"], default="auto",
                    help="GPU to compare against CPU (auto prefers CUDA, then MPS)")
    a = ap.parse_args()
    if a.repeats < 1:
        ap.error("--repeats must be positive")
    gpu = str(resolve_device(a.device))
    if gpu == "cpu":
        ap.error("No supported GPU available; install the appropriate PyTorch build or run on GPU hardware.")
    PHYSICS.clear()
    trips = load_kaggle(a.data)
    # Warm up both backends before measuring; synchronizing includes queued GPU work.
    D = dict(x=torch.zeros(32, 12, 9), m=torch.ones(32, 9), ctx=torch.zeros(32, 12))
    for device in ("cpu", gpu):
        net = VirtualSensorNet().to(device)
        fit(net, D, np.arange(24), np.arange(24, 32), 1, 1e-3)
        channel_errors(net, D)
        sync(device)
    rows = []
    for repeat in range(a.repeats):
        for device in (("cpu", gpu) if repeat % 2 == 0 else (gpu, "cpu")):
            args = argparse.Namespace(device=device, epochs=3, ft_epochs=2)
            torch.manual_seed(42)
            sync(device)
            start = time.perf_counter()
            P = prepare_car(a.car, trips, args)
            if P is None:
                raise RuntimeError(f"Not enough data for {a.car}")
            sync(device)
            trained = time.perf_counter()
            errors = channel_errors(P["tuned"], P["Dclean"])
            sync(device)
            end = time.perf_counter()
            row = dict(device=device, repeat=repeat + 1, train_seconds=trained-start,
                       inference_seconds=end-trained, total_seconds=end-start,
                       finite_errors=int(np.isfinite(errors).sum()))
            rows.append(row)
            print(json.dumps(row), flush=True)
            del P
    medians = {d: float(np.median([r["total_seconds"] for r in rows if r["device"] == d]))
               for d in ("cpu", gpu)}
    report = dict(gpu=gpu, car=a.car, epochs=3, ft_epochs=2, seed=42, torch=torch.__version__,
                  runs=rows, median_seconds=medians, speedup=medians["cpu"]/medians[gpu])
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
