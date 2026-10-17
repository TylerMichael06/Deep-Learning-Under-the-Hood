"""Thermostat audit, injection and score regression tests; no neural-network training."""
import unittest

import numpy as np
import pandas as pd

from obdfault.config import CH, DT, W
from obdfault.data import resample
from obdfault.features import PHYSICS, windows
from obdfault.thermostat import thermostat_audit, period_decision, combine_early_decision
from obdfault.scoring import WarmupRise
from obdfault.faults import delay_coolant


def trip(order=0, runtime_source="engine_runtime", runtime0=0.):
    n = 200
    X = np.tile([1500., 40., 30., 20., 30., 25., 50., 10., 0.], (n, 1))
    X[:, CH.index("ECT")] = np.minimum(30 + np.arange(n) * .5, 90)
    return dict(unit="example", order=order, X=X, ctx=runtime0 + np.arange(n) * DT / 60,
                code=np.zeros(n, bool), runtime_source=runtime_source)


class ThermostatTests(unittest.TestCase):
    def test_runtime_provenance_does_not_change_model_arrays(self):
        tr = trip()
        raw = pd.DataFrame(tr["X"], columns=CH)
        raw["t"] = np.arange(len(raw)) * DT
        raw["runtime"] = raw.t
        raw["code"] = 0.
        observed = resample(raw)
        raw["runtime"] = np.nan
        missing = resample(raw)
        self.assertEqual(observed["runtime_source"], "engine_runtime")
        self.assertEqual(missing["runtime_source"], "segment_elapsed")
        for key in ("X", "ctx", "code"):
            np.testing.assert_array_equal(observed[key], missing[key])

    def test_unknown_runtime_never_confirms_an_engine_start(self):
        for source in ("segment_elapsed", "unknown"):
            row = thermostat_audit([trip(runtime_source=source)]).iloc[0]
            self.assertTrue(row.cold_candidate)
            self.assertFalse(row.confirmed_start_candidate)
        self.assertTrue(thermostat_audit([trip()]).iloc[0].confirmed_start_candidate)

    def test_late_invalid_warm_coded_and_short_recordings_are_not_candidates(self):
        late, invalid, warm, coded, short = trip(runtime0=12), trip(runtime0=-1), trip(), trip(), trip()
        warm["X"][:, CH.index("ECT")] = 90
        coded["code"][80] = True
        for key in ("X", "ctx", "code"):
            short[key] = short[key][:60]
        report = thermostat_audit([late, invalid, warm, coded, short])
        self.assertFalse(report.cold_candidate.any())
        self.assertTrue(report.iloc[2].legacy_unchanged)

    def test_window_counts_match_pipeline_with_missing_data_and_codes(self):
        tr = trip()
        tr["X"][20:45, :6] = np.nan
        tr["code"][90:95] = True
        saved = PHYSICS[:]
        try:
            PHYSICS.clear()
            D = windows([tr], np.zeros(len(CH)), np.ones(len(CH)), stride=2)
        finally:
            PHYSICS[:] = saved
        row = thermostat_audit([tr]).iloc[0]
        self.assertEqual(row.windows, len(D["x"]))
        self.assertEqual(row.clean_windows, (~D["code"]).sum())
        self.assertEqual(row.judged_windows, (((D["start"] + W) * DT / 60) >= 5).sum())

    def test_split_keeps_warm_recordings_and_uses_time_not_candidate_count(self):
        trips = [trip(order=i) for i in range(10)]
        for tr in trips[:5]:
            tr["X"][:, CH.index("ECT")] = 90
        report = thermostat_audit(list(reversed(trips))).sort_values("order")
        self.assertEqual(report.split.tolist(), ["adaptation"] * 4 + ["calibration"] * 2 + ["test"] * 4)
        self.assertTrue(report.baseline_evaluable.all())
        self.assertEqual(report.groupby("split").cold_candidate.sum().to_dict(),
                         dict(adaptation=0, calibration=1, test=4))

    def test_absent_ect_and_excluded_car_are_not_candidates(self):
        missing, excluded = trip(), trip()
        missing["X"][:, CH.index("ECT")] = np.nan
        excluded["unit"] = "car6"
        report = thermostat_audit([missing, excluded])
        self.assertFalse(report.cold_candidate.any())
        self.assertFalse(report.confirmed_start_candidate.any())
        self.assertFalse(report.legacy_unchanged.any())

    def test_delayed_curve_preserves_missingness_other_sensors_and_input(self):
        tr = trip(); tr["X"][30:32, CH.index("ECT")] = np.nan
        before = tr["X"].copy()
        changed = delay_coolant(tr, .5)
        np.testing.assert_array_equal(tr["X"], before)
        np.testing.assert_array_equal(np.isnan(changed["X"]), np.isnan(before))
        other = [j for j, c in enumerate(CH) if c != "ECT"]
        np.testing.assert_array_equal(changed["X"][:, other], before[:, other])
        self.assertEqual(changed["X"][100, CH.index("ECT")], before[50, CH.index("ECT")])
        with self.assertRaises(ValueError):
            delay_coolant(tr, 1.)

    def test_delay_does_not_bridge_long_gaps_or_invent_a_warm_fault(self):
        tr = trip(); tr["X"][20:40, CH.index("ECT")] = np.nan
        changed = delay_coolant(tr, .5)
        self.assertEqual(changed["X"][60, CH.index("ECT")], tr["X"][60, CH.index("ECT")])
        tr["X"][:, CH.index("ECT")] = 90
        np.testing.assert_array_equal(delay_coolant(tr, .5)["X"], tr["X"])

    def test_rise_score_is_causal_and_does_not_refit_on_scoring(self):
        model = WarmupRise().fit([trip(i) for i in range(3)])
        tr = trip(); before = model.score(tr); coef = model.coef.copy()
        tr["X"][90:, :] *= 1.5
        after = model.score(tr)
        np.testing.assert_allclose(before[:90], after[:90], equal_nan=True)
        np.testing.assert_array_equal(coef, model.coef)

    def test_rise_abstains_for_missing_heat_inputs_warm_or_late_start(self):
        model = WarmupRise().fit([trip(i) for i in range(3)])
        missing, warm, late = trip(), trip(), trip(runtime0=12)
        missing["X"][:, CH.index("LOAD")] = np.nan
        warm["X"][:, CH.index("ECT")] = 90
        for tr in (missing, warm, late):
            self.assertTrue(np.isnan(model.score(tr)).all())
        with self.assertRaises(ValueError):
            WarmupRise().fit([trip(), trip()])

    def test_delayed_rise_produces_larger_deficit_on_known_artificial_curve(self):
        model = WarmupRise().fit([trip(i) for i in range(3)])
        normal = model.score(trip()); delayed = model.score(delay_coolant(trip(), .5))
        self.assertGreater(np.nanmean(delayed), np.nanmean(normal))

    def test_early_period_excludes_warmup_outside_its_own_interval(self):
        minute = np.array([1., 2., 2.2, 2.4, 3., 3.2, 4., 5., 6.])
        score = np.array([100., 0., 0., 0., 0., 0., 0., 100., 100.])
        d = period_decision(score, minute, 1.)
        self.assertEqual(d["eligible_windows"], 6)
        self.assertEqual(d["decision"], "HEALTHY")
        score[1:3] = 2.
        self.assertEqual(period_decision(score, minute, 1.)["decision"], "FAULT")

    def test_early_period_needs_six_finite_windows_and_uses_eligible_denominator(self):
        minute = 1 + np.arange(30) / 6
        score = np.full(30, np.nan); score[6:12] = 2.
        d = period_decision(score, minute, 1.)
        self.assertEqual(d["decision"], "FAULT")
        self.assertEqual(d["flagged_share"], 1.)
        score[6] = np.nan
        self.assertEqual(period_decision(score, minute, 1.)["decision"], "NO_CALL")

    def test_early_combination_preserves_baseline_and_abstention(self):
        healthy = dict(judged=True, decision="HEALTHY", first_alarm_min=np.nan, flagged_share=0.)
        late_fault = dict(healthy, decision="FAULT", first_alarm_min=6.)
        early_fault = dict(healthy, decision="FAULT", first_alarm_min=3.)
        unknown = dict(healthy, judged=False, decision="NO_CALL")
        self.assertEqual(combine_early_decision(healthy, early_fault)["decision"], "FAULT")
        self.assertEqual(combine_early_decision(late_fault, unknown)["decision"], "FAULT")
        self.assertEqual(combine_early_decision(unknown, healthy)["decision"], "NO_CALL")
        self.assertEqual(combine_early_decision(healthy, unknown)["decision"], "HEALTHY")
        self.assertEqual(combine_early_decision(late_fault, early_fault)["first_alarm_min"], 3.)
        self.assertEqual(healthy["decision"], "HEALTHY")


if __name__ == "__main__":
    unittest.main()
