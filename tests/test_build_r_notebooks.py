import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

from build_r_notebooks import (INPUT_DIR, MARKDOWN_SLUG, MARKDOWN_TITLE, NOTEBOOK_SLUG,  # noqa: E402
                               NOTEBOOK_TITLE, build)
from test_build_dataset import staged  # noqa: E402, F401  (fixture: a staged results dataset)

needs_r = pytest.mark.skipif(shutil.which("Rscript") is None, reason="R is not installed")


def _slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def test_metadata(tmp_path):
    build(tmp_path, owner="someone", public=True)

    nb = json.loads((tmp_path / "notebook" / "kernel-metadata.json").read_text())
    md = json.loads((tmp_path / "markdown" / "kernel-metadata.json").read_text())
    assert (nb["language"], nb["kernel_type"], nb["id"]) == ("r", "notebook", f"someone/{NOTEBOOK_SLUG}")
    assert (md["language"], md["kernel_type"], md["id"]) == ("rmarkdown", "script", f"someone/{MARKDOWN_SLUG}")
    assert nb["dataset_sources"] == md["dataset_sources"] == ["someone/asl-realtime-experiment-results"]
    assert nb["is_private"] == md["is_private"] == "false"
    assert (_slug(NOTEBOOK_TITLE), _slug(MARKDOWN_TITLE)) == (NOTEBOOK_SLUG, MARKDOWN_SLUG)
    assert "OWNER" not in (tmp_path / "markdown" / md["code_file"]).read_text()


@needs_r
def test_r_notebook_runs_on_the_staged_dataset(tmp_path, staged):  # noqa: F811
    build(tmp_path, owner="someone")
    nb = json.loads((tmp_path / "notebook" / f"{NOTEBOOK_SLUG}.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    (tmp_path / "nb.R").write_text(code.replace(INPUT_DIR, str(staged)))

    subprocess.run(["Rscript", "nb.R"], cwd=tmp_path, check=True, capture_output=True)


@needs_r
def test_r_markdown_knits_on_the_staged_dataset(tmp_path, staged):  # noqa: F811
    build(tmp_path, owner="someone")
    rmd = tmp_path / "markdown" / f"{MARKDOWN_SLUG}.Rmd"
    rmd.write_text(rmd.read_text().replace(INPUT_DIR, str(staged)))

    subprocess.run(["Rscript", "-e", f'knitr::knit("{rmd.name}", quiet = TRUE)'], cwd=rmd.parent,
                   check=True, capture_output=True)

    assert "tflite_int8" in (rmd.parent / f"{MARKDOWN_SLUG}.md").read_text()
