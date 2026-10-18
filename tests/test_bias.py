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


from obdfault.scoring import cusum, bias_scorer


def test_cusum_grows_steadily_on_a_constant_offset():
    out = cusum(np.full((20, 1), 1.0), k=0.5)
    assert np.allclose(out[:, 0], 0.5 * np.arange(1, 21))


def test_cusum_catches_both_directions():
    out = cusum(np.full((10, 2), [-1.0, 1.0]), k=0.5)          # sensor 0 reads high, sensor 1 reads low
    assert np.allclose(out[-1], [5.0, 5.0])


def test_cusum_stays_small_on_zero_mean_noise():
    z = np.random.RandomState(0).normal(0, 1, (500, 1))
    assert cusum(z, k=0.5).max() < 10                          # a constant offset of 1 would reach 250


def test_cusum_starts_growing_only_when_the_offset_starts():
    out = cusum(np.r_[np.zeros((10, 1)), np.ones((10, 1))], k=0.5)[:, 0]
    assert (out[:10] == 0).all() and np.isclose(out[-1], 5.0)


def test_cusum_skips_missing_values():
    out = cusum(np.array([[1.0], [np.nan], [1.0]]), k=0.5)[:, 0]
    assert np.isclose(out[0], 0.5) and np.isnan(out[1]) and np.isclose(out[2], 1.0)


def _two_drives(n=200):
    starts = np.arange(0, n, 2)[: n // 2]                       # one window every 10 s, as windows(stride=2) makes them
    return dict(trip=np.repeat([0, 1], n // 2), start=np.r_[starts, starts])


def test_bias_scorer_flags_an_offset_and_blames_its_sensor():
    rng, D = np.random.RandomState(0), _two_drives()
    B_cal = rng.normal(0.3, 1, (200, 2))                        # this car's model is always 0.3 off: calibrated away
    score, s_cal = bias_scorer(B_cal, D, k=0.5, grace=0)
    clean = rng.normal(0.3, 1, (200, 2)); faulty = clean.copy(); faulty[:, 1] += 1.5
    s_clean, _ = score(clean, D); s_fault, who = score(faulty, D)
    assert s_cal.shape == (200,) and s_clean.max() < 10
    assert s_fault.max() > 5 * s_clean.max()
    assert who[D["trip"] == 0][-1] == 1


def test_bias_scorer_waits_for_the_warm_up():
    D = _two_drives(); B = np.random.RandomState(0).normal(0, 1, (200, 2)); B[:, 1] += 3
    score, _ = bias_scorer(B, D, k=0.5, grace=5)
    s, who = score(B, D); early = D["start"] < 48               # window ends before minute 5: (start + 12) * 5 s < 300 s
    assert (s[early] == 0).all() and (who[early] == -1).all()
