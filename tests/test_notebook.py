import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_notebook import DATASET, KERNEL_SLUG, build  # noqa: E402


def _cells(nb, kind):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def test_build_writes_notebook_and_metadata(tmp_path):
    build(tmp_path, owner="someone")

    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == f"someone/{KERNEL_SLUG}"
    assert meta["dataset_sources"] == [DATASET]
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true"
    nb = json.loads((tmp_path / meta["code_file"]).read_text())
    assert nb["nbformat"] == 4


def test_code_cells_compile_and_have_no_relative_imports(tmp_path):
    build(tmp_path, owner="someone")
    nb = json.loads((tmp_path / f"{KERNEL_SLUG}.ipynb").read_text())

    for i, src in enumerate(_cells(nb, "code")):
        compile(src, f"cell{i}", "exec")
        assert "from ." not in src
        assert "__main__" not in src


def test_inlined_modules_define_a_working_pipeline(tmp_path):
    build(tmp_path, owner="someone")
    nb = json.loads((tmp_path / f"{KERNEL_SLUG}.ipynb").read_text())
    module_cells = ["".join(c["source"]) for c in nb["cells"]
                    if "module" in c.get("metadata", {}).get("tags", [])]

    ns: dict = {}
    for src in module_cells:
        exec(compile(src, "module", "exec"), ns)

    assert callable(ns["run"])
    assert ns["TrainConfig"]().model == "gru"
    assert ns["N_FEATURES"] == 132
