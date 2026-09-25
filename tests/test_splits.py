import numpy as np
import pytest

from obdfault.splits import (
    block_split_by_class,
    contiguous_split,
    group_split,
    leave_one_group_out,
    make_fold,
)


def test_contiguous_split_is_ordered_blocks():
    train, val, test = contiguous_split(100)
    assert train.tolist() == list(range(70))
    assert val.tolist() == list(range(70, 85))
    assert test.tolist() == list(range(85, 100))


def test_block_split_by_class_keeps_each_class_in_time_order():
    labels = np.array([0] * 20 + [1] * 40)
    train, val, test = block_split_by_class(labels)
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(60))
    assert train.tolist() == list(range(14)) + list(range(20, 48))
    assert val.tolist() == list(range(14, 17)) + list(range(48, 54))
    assert test.tolist() == list(range(17, 20)) + list(range(54, 60))


def test_group_split_has_no_group_overlap_and_is_deterministic():
    groups = np.repeat([f"g{i}" for i in range(10)], 5)
    train, val, test = group_split(groups, seed=0)
    sets = [set(groups[i]) for i in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(sets[1]) == 2 and len(sets[2]) == 2
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(50))
    assert [a.tolist() for a in group_split(groups, seed=0)] == [train.tolist(), val.tolist(), test.tolist()]


def test_group_split_needs_three_groups():
    with pytest.raises(ValueError):
        group_split(np.array(["a", "a", "b"]))


def test_leave_one_group_out_holds_out_each_group_and_uses_tails_for_val():
    groups = np.repeat(["a", "b", "c"], 20)
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    train, val, test = folds[0]
    assert set(groups[test]) == {"a"}
    assert "a" not in set(groups[train]) | set(groups[val])
    assert val.tolist() == list(range(37, 40)) + list(range(57, 60))
    assert train.tolist() == list(range(20, 37)) + list(range(40, 57))


def test_make_fold_slices_arrays():
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])
    groups = np.array(list("abcde"))
    fold = make_fold(X, y, groups, np.array([0, 1]), np.array([2]), np.array([3, 4]))
    assert fold.X_train.shape == (2, 2)
    assert fold.y_val.tolist() == [0]
    assert fold.test_groups.tolist() == ["d", "e"]
