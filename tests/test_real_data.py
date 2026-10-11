import numpy as np
import pandas as pd
from obdfault.config import CH
from obdfault.real_data import COLUMNS, load_bluedriver


def test_bluedriver_encoding_mapping_masks_and_context(tmp_path):
    frame = pd.DataFrame({'Time(s)': np.arange(121)})
    for c, value in [('RPM', 900), ('SPEED', 20), ('LOAD', 30), ('THROTTLE', 25), ('ECT', 80), ('IAT', 30), ('MAF', 8)]:
        frame[COLUMNS[c]] = value
    frame.loc[0, COLUMNS['RPM']] = 16383
    path = tmp_path / 'log.csv'
    path.write_text('BlueDriver Data Log\nOct 11, 2026\n' + frame.to_csv(index=False), encoding='utf-16')
    trips, report = load_bluedriver(path)
    assert report['sensors']['RPM']['out_of_range_rows'] == 1
    assert report['sensors']['STFT1']['column_present'] is False
    assert report['ground_truth_available'] is False
    assert len(trips) == 1
    assert np.isnan(trips[0]['X'][:, CH.index('STFT1')]).all()
    assert np.isnan(trips[0]['X'][:, CH.index('MAP')]).all()
    assert (trips[0]['ctx'] == 30).all()
    trips, _ = load_bluedriver(path, runtime_offset=60)
    assert trips[0]['ctx'][0] == 1


def test_bluedriver_splits_reset_without_joining_logs(tmp_path):
    frame = pd.DataFrame({'Time(s)': np.tile(np.arange(121), 2)})
    for c in ('RPM', 'SPEED', 'LOAD', 'THROTTLE'):
        frame[COLUMNS[c]] = 20
    path = tmp_path / 'log.csv'
    path.write_text('BlueDriver Data Log\ndate\n' + frame.to_csv(index=False))
    trips, report = load_bluedriver(path)
    assert len(trips) == report['usable_segments'] == 2
