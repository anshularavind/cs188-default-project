"""
Step 99: Colab Setup
====================
Bootstraps this project in Google Colab without creating a local venv.

Usage (inside Colab after cloning this repo):
    python cabinet_door_project/99_colab_setup.py

Optional:
    python cabinet_door_project/99_colab_setup.py --download_assets
    python cabinet_door_project/99_colab_setup.py --download_dataset
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def run_apt_install(packages: list[str]):
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    run(prefix + ["apt-get", "update", "-y"])
    run(prefix + ["apt-get", "install", "-y"] + packages)


def pip_install(args: list[str]):
    run([sys.executable, "-m", "pip", "install"] + args)


def ensure_repo(repo_root: Path):
    rs_dir = repo_root / "robosuite"
    rc_dir = repo_root / "robocasa"

    if not rs_dir.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/ARISE-Initiative/robosuite.git", str(rs_dir)])
    if not rc_dir.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/robocasa/robocasa.git", str(rc_dir)])

    return rs_dir, rc_dir


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CS188 cabinet project in Colab")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Repo root path (defaults to parent of this script)",
    )
    parser.add_argument(
        "--skip_apt",
        action="store_true",
        help="Skip apt package installation",
    )
    parser.add_argument(
        "--download_assets",
        action="store_true",
        help="Download RoboCasa kitchen assets after install",
    )
    parser.add_argument(
        "--download_dataset",
        action="store_true",
        help="Run 04_download_dataset.py after install",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run 00_verify_installation.py at the end",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    print("=" * 60)
    print("  OpenCabinet - Colab Setup")
    print("=" * 60)
    print(f"Repo root: {repo_root}")

    if not args.skip_apt:
        run_apt_install(
            [
                "git",
                "ffmpeg",
                "libgl1",
                "libglew2.2",
                "libosmesa6",
                "python3-dev",
                "build-essential",
            ]
        )

    pip_install(["--upgrade", "pip", "setuptools", "wheel"])

    rs_dir, rc_dir = ensure_repo(repo_root)

    # Core project dependencies.
    pip_install(["-e", str(rs_dir)])
    pip_install(["-e", str(rc_dir)])
    pip_install([
        "numpy==2.2.5",
        "mujoco==3.3.1",
        "pyarrow",
        "imageio",
        "imageio-ffmpeg",
        "gymnasium",
        "termcolor",
    ])

    # Colab-friendly default for offscreen MuJoCo.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    if args.download_assets:
        run([sys.executable, "-m", "robocasa.scripts.download_kitchen_assets"])

    if args.download_dataset:
        run([sys.executable, str(repo_root / "cabinet_door_project" / "04_download_dataset.py")])

    if args.verify:
        run([sys.executable, str(repo_root / "cabinet_door_project" / "00_verify_installation.py")])

    print("\nSetup complete.")
    print("Next steps:")
    print(f"  1. cd {repo_root / 'cabinet_door_project'}")
    print("  2. python 00_verify_installation.py")
    print("  3. python 04_download_dataset.py")
    print("  4. python 05b_augment_handle_data.py")
    print("  5. python 06_train_policy.py --config configs/diffusion_policy.yaml")
    print("     (quick sanity) python 06_train_policy.py --config configs/diffusion_policy.yaml --epochs 5 --max_episodes 40 --batch_size 32")


if __name__ == "__main__":
    main()
