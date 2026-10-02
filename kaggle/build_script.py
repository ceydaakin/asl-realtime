"""Generate the Kaggle utility script: the whole asl_realtime package in one importable file.

Other notebooks add it under kernel_sources. A script pushed through the API is not importable by
name (that needs "Set as utility script" in the Kaggle editor), so when it runs it also saves
itself as an output file, which a notebook can put on sys.path.

    python kaggle/build_script.py --owner <kaggle-username> [--public]
    kaggle kernels push -p kaggle/script
"""

import argparse
import json
from pathlib import Path

from build_notebook import MODULE_ORDER, module_source

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SLUG = "asl-realtime-lib"
TITLE = "ASL Realtime Lib"
KEYWORDS = ["classification", "transformers"]
MODULE = SCRIPT_SLUG.replace("-", "_")

HEADER = '''"""asl_realtime as a single module: landmark layout, normalization, augmentation, models, training.

Generated from https://github.com/ceydaakin/asl-realtime. Add this script to a notebook as a
utility script, then:

    from asl_realtime_lib import TrainConfig, run, load_model
"""
'''


FOOTER = f'''
if __name__ == "__main__":  # the Kaggle run: publish this file as an output other notebooks can import
    import shutil
    import sys

    shutil.copy(globals().get("__file__", sys.argv[0]), "{MODULE}.py")
'''


def source() -> str:
    return HEADER + "\n" + "\n\n".join(module_source(m) for m in MODULE_ORDER) + FOOTER


def build(out_dir: Path, owner: str, public: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{SCRIPT_SLUG}.py"
    (out_dir / code_file).write_text(source())
    meta = {
        "id": f"{owner}/{SCRIPT_SLUG}", "title": TITLE, "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": str(not public).lower(),
        "enable_gpu": "false", "enable_internet": "false",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [], "keywords": KEYWORDS,
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "script")
    p.add_argument("--public", action="store_true", help="publish the script publicly")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner, a.public)}")


if __name__ == "__main__":
    main()
