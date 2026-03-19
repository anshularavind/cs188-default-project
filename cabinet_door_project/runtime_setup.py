"""Runtime setup helpers for cross-platform MuJoCo execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_colab() -> bool:
    if "COLAB_GPU" in os.environ:
        return True
    if "google.colab" in sys.modules:
        return True
    # Heuristic for notebook VMs where module probing may fail in subprocesses.
    return Path("/content").exists() and "COLAB_RELEASE_TAG" in os.environ


def is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        text = Path("/proc/version").read_text().lower()
    except Exception:
        return False
    return "microsoft" in text or "wsl" in text


def configure_offscreen_gl() -> str:
    """
    Configure a sane Linux offscreen backend.

    Policy:
    - Colab / non-WSL Linux: EGL (GPU-capable and usually available)
    - WSL: OSMesa (EGL typically cannot access /dev/dri)

    Returns:
        backend string: "egl", "osmesa", or "unchanged"
    """
    if sys.platform != "linux":
        return "unchanged"

    if is_wsl():
        backend = "osmesa"
    else:
        backend = "egl"

    os.environ.setdefault("MUJOCO_GL", backend)
    os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    return backend


def select_torch_device():
    """Pick best available torch device with CUDA > MPS > CPU priority."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
