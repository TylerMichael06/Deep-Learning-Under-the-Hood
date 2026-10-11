"""Drive decisions count only windows outside the warm-up period."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from obdfault import evaluate
from obdfault.config import CH, DT, W
from obdfault.synthetic import decide


@pytest.mark.parametrize("judged, flagged, expected", [
    (6, 3, "FAULT"),
    (10, 3, "HEALTHY"),  # exactly 30% does not cross the strict cutoff
    (10, 4, "FAULT"),
    (5, 5, "TOO SHORT"),
    (0, 0, "TOO SHORT"),
])
def test_synthetic_drive_ignores_warmup(judged, flagged, expected):
    n = 24 + judged
    rows = pd.DataFrame({
        "car": "car1", "drive": 0,
        "minute": np.r_[np.arange(24) / 6 + 1, 5 + np.arange(judged) / 6],
        "score": np.r_[np.full(24, 2), np.full(flagged, 2), np.zeros(judged - flagged)],
        "blamed": "ECT", "truth": True, "thr_5": np.ones(n),
    })
    key = pd.DataFrame({
        "car": ["car1", "car2"], "drive": [0, 0], "start_min": [0, 0],
    })
    scored, verdicts = decide(rows, key, 5, 0.3)
    assert not scored.loc[~scored.judged, "flagged"].any()
    assert verdicts.model_says.tolist() == [expected, "TOO SHORT"]


def test_main_evaluation_counts_only_judged_windows(monkeypatch):
    # One 50%-flagged drive, one exactly-30% drive, one short drive,
    # and one drive entirely in warm-up. Both clean and faulty copies use these
    # scores to verify both sides of the drive-level confusion matrix.
    trip, minute, score = [], [], []
    for drive, (judged, flagged) in enumerate([(6, 3), (10, 3), (5, 5), (0, 0)]):
        trip.extend([drive] * (24 + judged))
        minute.extend(np.r_[np.arange(24) / 6 + 1, 5 + np.arange(judged) / 6])
        score.extend([2] * 24 + [2] * flagged + [0] * (judged - flagged))
    score = np.array(score)
    data = dict(trip=np.array(trip), start=np.array(minute) * 60 / DT - W,
                code=np.zeros(len(trip), bool), m=torch.ones((len(trip), len(CH))))
    prepared = dict(base=None, tuned=None, mu=None, sd=None, Dtr=data,
                    Dcal=data, Dclean=data, test_trips=[], cal_trips=[])
    monkeypatch.setattr(evaluate, "prepare_car", lambda *args: prepared)
    monkeypatch.setattr(evaluate, "ALL_FAULTS", {"test fault": ("ECT",)})
    monkeypatch.setattr(evaluate, "inject", lambda *args: [])
    monkeypatch.setattr(evaluate, "windows", lambda *args, **kwargs: data)
    monkeypatch.setattr(evaluate, "channel_errors", lambda *args: score)
    monkeypatch.setattr(evaluate, "ridge_errors", lambda fit, datasets: [score] * len(datasets))
    who = np.full(len(score), CH.index("ECT"))
    monkeypatch.setattr(evaluate, "calibrate", lambda *args: (lambda *args: (score, who), np.ones(len(score))))
    monkeypatch.setattr(evaluate, "slow_stuck_scorer", lambda *args, **kwargs: (lambda *args: (score, who), np.ones(len(score))))
    args = SimpleNamespace(grace=5, min_judge=6, cutoffs=[0.9], drive_rules=[0.3], smooth=12)

    results, summary = evaluate.evaluate_car("car1", [], args)

    assert (results.drive_TP == 1).all()
    assert (results.drive_FN == 1).all()
    assert (results.drive_FP == 1).all()
    assert (results.drive_TN == 1).all()
    assert (results.drives_too_short == 2).all()
    assert (summary.clean_drives_judged == 2).all()
    assert (summary.clean_drives_too_short == 2).all()
