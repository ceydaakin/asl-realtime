import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_dataset import DATASET_SLUG, RUNS, SCHEMA, build  # noqa: E402

EPOCH = {"train_loss": 2.0, "val_loss": 1.5, "val_top1": 0.5, "val_top5": 0.8, "seconds": 30.0}


def _metrics(old_format: bool) -> dict:
    m = {"model": "conv1d", "params": 10, "n_train": 8, "n_val": 2, "device": "mps", "epochs": 2,
         "best_epoch": 2, "val_top1": 0.5, "val_top5": 0.8, "last_val_top1": 0.5, "last_val_top5": 0.8,
         "history": [{"epoch": 1, **EPOCH}, {"epoch": 2, **EPOCH}]}
    return m if old_format else {**m, "norm": "sequence", "augment": True}


@pytest.fixture
def staged(tmp_path):
    runs, exports = tmp_path / "runs", tmp_path / "exports"
    exports.mkdir()
    for name in RUNS:
        (runs / name).mkdir(parents=True)
        (runs / name / "metrics.json").write_text(json.dumps(_metrics(old_format=name == "gru")))
    (exports / "signs.json").write_text('["TV", "after"]')
    (exports / "report.json").write_text(json.dumps({"variants": {"tflite_int8": {
        "file": "asl_int8.tflite", "size_mb": 1.5, "top1": 0.74, "torch_top1": 0.75,
        "agreement": 0.99, "max_prob_diff": 0.1, "host_ms_median": 0.5}}}))
    return build(tmp_path / "out", "someone", runs, exports)


def _rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def test_every_file_matches_its_documented_columns(staged):
    meta = json.loads((staged / "dataset-metadata.json").read_text())

    assert meta["id"] == f"someone/{DATASET_SLUG}" and meta["isPrivate"] is True
    assert meta["subtitle"] and meta["keywords"] and meta["licenses"]
    assert meta["userSpecifiedSources"] and meta["expectedUpdateFrequency"] == "never"
    assert (staged / meta["image"]).is_file()
    assert {r["path"] for r in meta["resources"]} == set(SCHEMA) == {p.name for p in staged.glob("*.csv")}
    for resource in meta["resources"]:
        rows = _rows(staged / resource["path"])
        assert rows and resource["description"]
        assert list(rows[0]) == [f["name"] for f in resource["schema"]["fields"]]
        assert all(f["description"] for f in resource["schema"]["fields"])
        assert all(v != "" for row in rows for v in row.values())


def test_runs_and_epochs(staged):
    runs = {r["run"]: r for r in _rows(staged / "runs.csv")}

    assert set(runs) == set(RUNS)
    assert (runs["gru"]["norm"], runs["gru"]["augment"]) == ("global", "False")  # pre-norm metrics format
    assert runs["conv1d_seq_taug"]["temporal_augment"] == "True"
    assert len(_rows(staged / "epochs.csv")) == 2 * len(RUNS)
    assert _rows(staged / "signs.csv")[1] == {"index": "1", "sign": "after"}
