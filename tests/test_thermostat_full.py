"""Check full-audit verdict parity and reuse of the official synthetic scenarios."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from synthdata import make_trips, EVAL_ARGS, SYN_ARGS
from obdfault import synthetic
from obdfault.features import PHYSICS
from obdfault.train import prepare_car
from obdfault.model import channel_errors
from obdfault.scoring import slow_stuck_scorer
from obdfault.config import DT, W
from obdfault.thermostat import thermostat_audit

spec = importlib.util.spec_from_file_location('thermostat_full', Path(__file__).parents[1]/'scripts/compare_thermostat_full.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_full_verdict_matches_official_synthetic_rule():
    # Includes a boundary at exactly 30%, a short drive, and warm-up dilution.
    for n, late, flagged in [(20, 10, 6), (20, 10, 7), (20, 5, 5), (30, 8, 8)]:
        minute = np.r_[np.linspace(1, 4.9, n-late), np.arange(late)+5.]
        score = np.zeros(n); score[-flagged:] = 2.
        rows = pd.DataFrame(dict(car='x', drive=0, minute=minute, score_only=score,
            blamed_only='ECT', thr_only_5=1., truth=True))
        key = pd.DataFrame(dict(car=['x'], drive=[0], start_min=[0.]))
        _, official = synthetic.decide(rows, key, 5, .3, 'only')
        judged, alarm = audit_module.verdict(score, minute, 1., np.ones(n, bool))
        assert judged == official.judged.iloc[0]
        assert alarm == (official.model_says.iloc[0] == 'FAULT')


def test_clean_verdict_excludes_coded_windows():
    minute = np.r_[np.arange(4)+1., np.arange(8)+5.]
    score = np.r_[np.zeros(4), np.full(8, 2.)]
    valid = np.r_[np.ones(6, bool), np.zeros(6, bool)]
    assert audit_module.verdict(score, minute, 1., valid) == (False, False)
    assert audit_module.verdict(score, minute, 1., np.ones(12, bool)) == (True, True)


def test_synthetic_assignment_matches_existing_suite(monkeypatch):
    PHYSICS.clear(); torch.set_num_threads(2); torch.manual_seed(0)
    trips = make_trips()
    P = prepare_car('car2', trips, EVAL_ARGS)
    monkeypatch.setattr(audit_module, 'SYNTHETIC_CARS', ['car2'])
    cases = audit_module.synthetic_cases(trips, thermostat_audit(trips))['car2']
    monkeypatch.setattr(synthetic, 'prepare_car', lambda *args: P)
    original_windows = synthetic.windows
    captured = []
    def capture(test, *args, **kwargs):
        captured.extend(test)
        return original_windows(test, *args, **kwargs)
    monkeypatch.setattr(synthetic, 'windows', capture)
    official_rows, key = synthetic.test_car('car2', trips, SYN_ARGS, np.random.RandomState(42))
    assert [case[0] for case in cases] == key.fault.tolist()
    for (_, case, _), official in zip(cases, captured):
        np.testing.assert_array_equal(case['X'], official['X'])

    # The audit uses the main evaluation scorer; it must also match the synthetic
    # suite's independently implemented final score and calibration threshold.
    scorer, calibration = slow_stuck_scorer(
        channel_errors(P['tuned'], P['Dcal']), P['Dcal'], P['cal_trips'], EVAL_ARGS, use_slow=False)
    D = original_windows(captured, P['mu'], P['sd'], stride=2)
    scores, _ = scorer(channel_errors(P['tuned'], D), D, captured)
    np.testing.assert_allclose(scores, official_rows.score_only, rtol=1e-12, atol=1e-12)
    late = (P['Dcal']['start'] + W) * DT / 60 >= 5
    np.testing.assert_allclose(np.quantile(calibration[late], .9), official_rows.thr_only_5, rtol=1e-12)
