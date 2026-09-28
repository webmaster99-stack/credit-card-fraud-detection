"""Stage the Hugging Face Space folder, and optionally upload it.

The Space runs `app.py` with the `fraud` package vendored next to it (`src/fraud`) and `params.yaml`
at the Space root, which keeps the layout `fraud.params.REPO_ROOT` expects. The model bundle is not
included: the app downloads it from the HF Hub model repo at startup.

Usage:
    uv run python demo/build_space.py            # stage data/space/
    uv run python demo/build_space.py --push     # ...and upload it to the Space repo
"""

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from fraud.params import REPO_ROOT, load_params

DEMO_DIR = REPO_ROOT / "demo"
STAGE_DIR = REPO_ROOT / "data" / "space"
SPACE_FILES = ["app.py", "README.md", "requirements.txt"]


def stage_space(out_dir: Path = STAGE_DIR) -> Path:
    """Copy exactly what the Space needs into ``out_dir`` (rebuilt from scratch each time)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in SPACE_FILES:
        shutil.copy2(DEMO_DIR / name, out_dir / name)
    shutil.copy2(REPO_ROOT / "params.yaml", out_dir / "params.yaml")
    shutil.copytree(
        REPO_ROOT / "src" / "fraud",
        out_dir / "src" / "fraud",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STAGE_DIR)
    parser.add_argument("--push", action="store_true", help="upload the folder to the Space repo")
    args = parser.parse_args()

    staged = stage_space(args.out)
    print(f"Staged Space files in {staged}")
    if args.push:
        from huggingface_hub import HfApi

        load_dotenv(REPO_ROOT / ".env")
        repo = load_params()["serving"]["hf_space_repo"]
        HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
            folder_path=str(staged),
            repo_id=repo,
            repo_type="space",
            commit_message="Update demo",
        )
        print(f"Pushed to https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
