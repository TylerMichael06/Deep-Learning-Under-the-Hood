"""Unit tests for the signed-error (bias) score, issue #33."""
import numpy as np, torch
from synthdata import make_trips
from obdfault.features import PHYSICS, windows
from obdfault.model import VirtualSensorNet, channel_errors, ridge_errors


def _windows():
    PHYSICS.clear(); torch.manual_seed(0)
    trips = make_trips(n_cars=1, n_trips=2)
    allX = np.concatenate([t["X"] for t in trips]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    return windows(trips, mu, sd, stride=2)


def _check_signed(E, E2, B):
    assert np.allclose(E, E2, equal_nan=True)                  # squared error unchanged
    assert np.array_equal(np.isnan(B), np.isnan(E))            # signed error exists exactly where the squared one does
    ok = ~np.isnan(E); assert (B[ok] ** 2 <= E[ok] + 1e-6).all()   # (mean error)^2 <= mean squared error


def test_channel_errors_can_also_return_the_signed_error():
    D = _windows(); model = VirtualSensorNet()
    E = channel_errors(model, D); E2, B = channel_errors(model, D, signed=True)
    _check_signed(E, E2, B)


def test_ridge_errors_can_also_return_the_signed_error():
    D = _windows()
    [E] = ridge_errors(D, [D]); [(E2, B)] = ridge_errors(D, [D], signed=True)
    _check_signed(E, E2, B)
