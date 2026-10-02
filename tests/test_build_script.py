import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_script import SCRIPT_SLUG, TITLE, build  # noqa: E402


def test_build_writes_script_metadata(tmp_path):
    build(tmp_path, owner="someone", public=True)

    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == f"someone/{SCRIPT_SLUG}"
    assert (meta["kernel_type"], meta["is_private"]) == ("script", "false")
    assert (tmp_path / meta["code_file"]).is_file()
    assert re.sub(r"[^a-z0-9]+", "-", TITLE.lower()).strip("-") == SCRIPT_SLUG


def test_script_is_one_importable_module(tmp_path):
    build(tmp_path, owner="someone")
    src = (tmp_path / f"{SCRIPT_SLUG}.py").read_text()

    assert "from ." not in src and "__main__" not in src
    ns: dict = {}
    exec(compile(src, "asl_realtime_lib", "exec"), ns)
    assert callable(ns["run"]) and callable(ns["load_model"])
    assert ns["TrainConfig"]().model == "gru"
