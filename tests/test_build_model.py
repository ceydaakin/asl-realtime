import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_model import MODEL_SLUG, VARIATIONS, build  # noqa: E402

METRICS = {"model": "transformer", "norm": "sequence", "params": 1262650, "epochs": 60,
           "n_train": 80229, "n_val": 14248, "val_top1": 0.7473, "val_top5": 0.9227}


@pytest.fixture
def staged(tmp_path):
    exports, run = tmp_path / "exports", tmp_path / "run"
    exports.mkdir()
    run.mkdir()
    (run / "model.pt").write_bytes(b"pt")
    (run / "metrics.json").write_text(json.dumps(METRICS))
    (exports / "signs.json").write_text('["TV", "after"]')
    variants = {}
    for v in VARIATIONS:
        if v.report_key is None:
            continue
        variants[v.report_key] = {"size_mb": 1.5, "top1": 0.746, "agreement": 0.99}
        if v.source.endswith(".mlpackage"):
            (exports / v.source / "Data").mkdir(parents=True)
            (exports / v.source / "Data" / "weight.bin").write_bytes(b"w")
        else:
            (exports / v.source).write_bytes(b"tfl")
    (exports / "report.json").write_text(json.dumps({"variants": variants}))
    return build(tmp_path / "out", "someone", exports, run)


def _instances(out):
    return {d.name: json.loads((d / "model-instance-metadata.json").read_text())
            for d in sorted((out / "variations").iterdir())}


def test_model_metadata_is_complete(staged):
    meta = json.loads((staged / "model-metadata.json").read_text())

    assert (meta["ownerSlug"], meta["slug"], meta["isPrivate"]) == ("someone", MODEL_SLUG, True)
    assert meta["subtitle"]
    for heading in ("# Model Summary", "# Model Characteristics", "# Data Overview", "# Evaluation Results"):
        assert heading in meta["description"]
    assert "74.7%" in meta["description"] and "1,262,650" in meta["description"]


def test_every_variation_has_metadata_weights_and_sign_names(staged):
    instances = _instances(staged)

    assert len(instances) == len(VARIATIONS)
    assert len({(m["framework"], m["instanceSlug"]) for m in instances.values()}) == len(VARIATIONS)
    for name, meta in instances.items():
        files = {p.name for p in (staged / "variations" / name).iterdir()}
        assert "signs.json" in files and len(files) == 3
        assert meta["modelSlug"] == MODEL_SLUG
        assert meta["licenseName"] and meta["overview"] and meta["trainingData"]
        assert "INSERT" not in json.dumps(meta)
        assert "# Model Inputs" in meta["usage"]


def test_mlpackage_directories_are_zipped(staged):
    archive = staged / "variations" / "other-coreml-fp16" / "asl_fp16.mlpackage.zip"

    assert zipfile.ZipFile(archive).namelist()[-1] == "asl_fp16.mlpackage/Data/weight.bin"


def test_public_build(staged, tmp_path):
    out = build(tmp_path / "out", "someone", tmp_path / "exports", tmp_path / "run", public=True)

    assert json.loads((out / "model-metadata.json").read_text())["isPrivate"] is False
