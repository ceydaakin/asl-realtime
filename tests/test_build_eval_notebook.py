import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_eval_notebook import KERNEL_SLUG, LIB, TITLE, build  # noqa: E402
from build_script import source  # noqa: E402


def _code(tmp_path):
    build(tmp_path, owner="someone")
    nb = json.loads((tmp_path / f"{KERNEL_SLUG}.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_metadata_links_script_model_and_datasets(tmp_path):
    build(tmp_path, owner="someone", public=True)

    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == f"someone/{KERNEL_SLUG}" and meta["is_private"] == "false"
    assert meta["kernel_sources"] == ["someone/asl-realtime-lib"]
    assert meta["model_sources"] == ["someone/asl-realtime-transformer/pyTorch/default/1"]
    assert "someone/asl-realtime-experiment-results" in meta["dataset_sources"]
    assert re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-") == KERNEL_SLUG


def test_cells_compile_and_import_only_names_the_library_defines(tmp_path):
    cells = _code(tmp_path)
    for i, src in enumerate(cells):
        compile(src, f"cell{i}", "exec")

    lib: dict = {}
    exec(compile(source(), LIB, "exec"), lib)
    imported = re.search(rf"from {LIB} import (.+)", cells[0]).group(1).split(", ")
    assert imported and all(name in lib for name in imported)
