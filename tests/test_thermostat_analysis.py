"""Guard the paired unit of analysis and unchanged full-drive denominator."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('thermostat_analysis',Path(__file__).parents[1]/'scripts/analyze_thermostat.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_pairing_rejects_duplicate_source_recordings():
    base=pd.DataFrame(dict(car=['car9'],trip_id=[1],scenario=['delay'],judged=[True],decision=['HEALTHY']))
    added=base.assign(decision='FAULT')
    result=analysis.paired_comparison(base,added)
    assert result['gains']==1 and result['n']==1
    with pytest.raises(pd.errors.MergeError):
        analysis.paired_comparison(base,pd.concat([added,added]))


def test_full_drive_keeps_suppressed_windows_and_no_call():
    # Eight late flags among 30 total windows: healthy under the official rule.
    minute=np.r_[np.linspace(1,4.9,22),np.linspace(5,6.2,8)]
    g=pd.DataFrame(dict(minute=minute,baseline_score=2.,baseline_threshold=1.,
        rise_score=np.nan,early_threshold=1.,original_rise_finite=False,code=False,cold=True,fault=False))
    d=analysis.full_drive_decision(g)
    assert d['judged'] and d['baseline']=='HEALTHY' and d['combined']=='HEALTHY'
    g.loc[g.minute.between(2,4.99),'rise_score']=2.
    g['original_rise_finite']=True
    assert analysis.full_drive_decision(g)['combined']=='FAULT'
    short=g.iloc[:-3]
    assert analysis.full_drive_decision(short)['combined']=='NO_CALL'
