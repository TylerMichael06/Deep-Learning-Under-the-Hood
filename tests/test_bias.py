"""Unit tests for the signed-error (bias) score, issue #33."""
import numpy as np, torch
from synthdata import make_trips
from obdfault.features import PHYSICS, windows
from obdfault.model import VirtualSensorNet, channel_errors, ridge_errors
from obdfault.config import CH


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


from obdfault.scoring import running_cusum, bias_scorer


def test_running_cusum_grows_steadily_on_a_constant_offset():
    out = running_cusum(np.full((20, 1), 1.0), k=0.5)
    assert np.allclose(out[:, 0], 0.5 * np.arange(1, 21))


def test_running_cusum_catches_both_directions():
    out = running_cusum(np.full((10, 2), [-1.0, 1.0]), k=0.5)          # sensor 0 reads high, sensor 1 reads low
    assert np.allclose(out[-1], [5.0, 5.0])


def test_running_cusum_stays_small_on_zero_mean_noise():
    z = np.random.RandomState(0).normal(0, 1, (500, 1))
    assert running_cusum(z, k=0.5).max() < 10                          # a constant offset of 1 would reach 250


def test_running_cusum_starts_growing_only_when_the_offset_starts():
    out = running_cusum(np.r_[np.zeros((10, 1)), np.ones((10, 1))], k=0.5)[:, 0]
    assert (out[:10] == 0).all() and np.isclose(out[-1], 5.0)


def test_running_cusum_skips_missing_values():
    out = running_cusum(np.array([[1.0], [np.nan], [1.0]]), k=0.5)[:, 0]
    assert np.isclose(out[0], 0.5) and np.isnan(out[1]) and np.isclose(out[2], 1.0)


from obdfault.scoring import BIAS_SENSORS

MAP, MAF, IAT, STFT = CH.index("MAP"), CH.index("MAF"), CH.index("IAT"), CH.index("STFT1")


def _drives(n=300, k=3):
    """k calibration drives of n // k windows each (bias_scorer needs at least 3)."""
    starts = np.arange(0, 2 * (n // k), 2)                      # one window every 10 s, as windows(stride=2) makes them
    return dict(trip=np.repeat(np.arange(k), n // k), start=np.tile(starts, k))


def test_bias_scorer_flags_an_offset_and_blames_its_sensor():
    rng, D = np.random.RandomState(0), _drives()
    B_cal = rng.normal(0.3, 1, (300, len(CH)))                  # this car's model is always 0.3 off: calibrated away
    score, s_cal = bias_scorer(B_cal, D, k=0.5, grace=0)
    clean = rng.normal(0.3, 1, (300, len(CH))); faulty = clean.copy(); faulty[:, MAP] += 1.5
    s_clean, _ = score(clean, D); s_fault, who = score(faulty, D)
    assert s_cal.shape == (300,) and s_clean.max() < 10
    assert s_fault.max() > 5 * s_clean.max()
    assert who[D["trip"] == 0][-1] == MAP


def test_bias_scorer_only_watches_the_air_sensors():
    assert BIAS_SENSORS == ["MAP", "LOAD"]
    rng, D = np.random.RandomState(0), _drives()
    score, _ = bias_scorer(rng.normal(0, 1, (300, len(CH))), D, k=0.5, grace=0)
    clean = rng.normal(0, 1, (300, len(CH))); off = clean.copy()
    off[:, [MAF, IAT, STFT]] += 3                              # MAF (drive gap check), intake temp, fuel trim: not watched
    s_clean, _ = score(clean, D); s_off, who = score(off, D)
    assert np.allclose(s_off, s_clean) and not np.isin(who, [MAF, IAT, STFT]).any()


def test_bias_scorer_calibration_scores_are_leave_one_drive_out():
    D = dict(trip=np.repeat([0, 1, 2], 100), start=np.tile(np.arange(0, 200, 2), 3))
    B = np.random.RandomState(0).normal(0, 1, (300, len(CH))); B[D["trip"] == 2, MAP] += 1.0   # one drive with its own offset
    score, s_cal = bias_scorer(B, D, k=0.5, grace=0)
    in_sample, _ = score(B, D); last = D["trip"] == 2
    assert s_cal[last].max() > 2 * in_sample[last].max()     # judged by the other drives, the offset stands out


def test_bias_scorer_is_off_with_fewer_than_three_calibration_drives():
    D = _drives(n=200, k=2); B = np.random.RandomState(0).normal(0, 1, (200, len(CH)))
    assert bias_scorer(B, D, k=0.5, grace=0) == (None, None)


def test_bias_scorer_waits_for_the_warm_up():
    D = _drives(); B = np.random.RandomState(0).normal(0, 1, (300, len(CH))); B[:, MAP] += 3
    score, _ = bias_scorer(B, D, k=0.5, grace=5)
    s, who = score(B, D); early = D["start"] < 48               # window ends before minute 5: (start + 12) * 5 s < 300 s
    assert (s[early] == 0).all() and (who[early] == -1).all() and s[~early].max() > 0


def test_slow_stuck_scorer_alarm_line_uses_leave_one_out_bias_scores():
    from synthdata import EVAL_ARGS
    from obdfault.scoring import slow_stuck_scorer, unit_scale
    from obdfault.config import DT, W
    PHYSICS.clear(); torch.manual_seed(0)
    trips = make_trips(n_cars=1, n_trips=3)
    allX = np.concatenate([t["X"] for t in trips]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    D = windows(trips, mu, sd, stride=2); E, B = channel_errors(VirtualSensorNet(), D, signed=True)
    _, s_plain = slow_stuck_scorer(E, D, trips, EVAL_ARGS, use_slow=False)
    _, s_bias = slow_stuck_scorer(E, D, trips, EVAL_ARGS, use_slow=False, B_cal=B)
    _, loo = bias_scorer(B, D, EVAL_ARGS.cusum_k, EVAL_ARGS.grace)
    warm = (D["start"] + W) * DT / 60 >= EVAL_ARGS.grace
    assert np.allclose(s_bias, np.maximum(s_plain, unit_scale(loo[warm], loo)))


def _three_drive_car(n_trips=3):
    from synthdata import EVAL_ARGS
    PHYSICS.clear(); torch.manual_seed(0)
    trips = make_trips(n_cars=1, n_trips=n_trips)
    allX = np.concatenate([t["X"] for t in trips]); mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    D = windows(trips, mu, sd, stride=2); E, B = channel_errors(VirtualSensorNet(), D, signed=True)
    return trips, D, E, B, EVAL_ARGS


def test_slow_stuck_scorer_falls_back_to_the_stuck_check_below_three_drives():
    from obdfault.scoring import slow_stuck_scorer
    trips, D, E, B, args = _three_drive_car(n_trips=2)
    plain, s_plain = slow_stuck_scorer(E, D, trips, args, use_slow=False)
    withb, s_withb = slow_stuck_scorer(E, D, trips, args, use_slow=False, B_cal=B)
    assert np.array_equal(s_plain, s_withb) and np.array_equal(plain(E, D, trips)[0], withb(E, D, trips, B)[0])


def test_slow_stuck_scorer_without_signed_errors_gives_the_regular_part():
    from obdfault.scoring import slow_stuck_scorer
    trips, D, E, B, args = _three_drive_car()
    plain, _ = slow_stuck_scorer(E, D, trips, args, use_slow=False)
    withb, _ = slow_stuck_scorer(E, D, trips, args, use_slow=False, B_cal=B)
    assert np.array_equal(withb(E, D, trips)[0], plain(E, D, trips)[0])
    assert (withb(E, D, trips, B)[0] >= plain(E, D, trips)[0]).all()


from obdfault.scoring import confirm, confirm_bias


def test_confirm_needs_two_of_the_last_three_drives():
    assert confirm([1, 0, 1, 1, 0]).tolist() == [False, False, True, True, False]
    assert confirm([1]).tolist() == [False]                    # the first drive is only "pending"
    assert confirm([1, 0, 0, 1]).tolist() == [False, False, False, False]


def test_confirm_bias_only_delays_alarms_the_regular_score_did_not_raise():
    regular = [0, 1, 0, 0, 0]; combined = [1, 1, 0, 1, 0]       # drive 2: regular score; drives 1 and 4: bias score only
    assert confirm_bias(regular, combined).tolist() == [False, True, False, True, False]


def test_with_gap_adds_maf_gap_alarms_and_keeps_drives_without_a_gap():
    import pandas as pd
    from obdfault.scoring import with_gap
    verdict = pd.Series([False, True, False, False], index=[0, 1, 2, 3])
    gap = pd.Series([True, False, False], index=[0, 1, 2])           # drive 3 has no gap value (e.g. too few MAF windows)
    assert with_gap(verdict, gap).tolist() == [True, True, False, False]
    assert with_gap(verdict, None).equals(verdict)                   # car without the gap check
