from pathlib import Path

import pandas as pd
import pytest

from plant_classifier.core import (
    build_class_mapping,
    discover_data,
    stratified_split,
    validate_submission,
)


def test_build_class_mapping_is_sorted(tmp_path: Path):
    train = tmp_path / "train"
    (train / "Sugar beet").mkdir(parents=True)
    (train / "Black-grass").mkdir()
    assert build_class_mapping(train) == {"Black-grass": 0, "Sugar beet": 1}


def test_discover_data_prefers_matching_template(tmp_path: Path):
    root = tmp_path / "nested" / "dataset"
    (root / "train" / "A").mkdir(parents=True)
    (root / "test").mkdir()
    (root / "test" / "x.png").write_bytes(b"x")
    pd.DataFrame({"ID": ["x.png"], "Category": ["A"]}).to_csv(
        tmp_path / "submission-for-task2.csv", index=False
    )
    found = discover_data(tmp_path)
    assert found.train_dir == root / "train"
    assert found.test_dir == root / "test"
    assert found.submission_template == tmp_path / "submission-for-task2.csv"


def test_stratified_split_preserves_all_classes():
    labels = [0] * 10 + [1] * 10 + [2] * 10
    train_idx, val_idx = stratified_split(labels, val_size=0.2, seed=42)
    assert {labels[i] for i in train_idx} == {0, 1, 2}
    assert {labels[i] for i in val_idx} == {0, 1, 2}
    assert set(train_idx).isdisjoint(val_idx)


def test_validate_submission_rejects_missing_id(tmp_path: Path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": ["a.png"], "Category": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ID set mismatch"):
        validate_submission(path, ["a.png", "b.png"], ["A", "B"])
