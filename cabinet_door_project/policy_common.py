"""
Shared policy utilities for OpenCabinet low-dimensional policies.

This module provides:
- Dataset loading that auto-uses 05b augmented parquet files when present
- A 1D Conv U-Net diffusion model for action-sequence generation
- Inference wrappers for both diffusion and legacy MLP checkpoints
- Online handle feature extraction so eval uses the same 05b feature schema
- Success check helper that treats opening any single cabinet door as success
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AUGMENTED_STATE_KEYS = [
    "observation.handle_pos",
    "observation.handle_to_eef_pos",
    "observation.door_openness",
    "observation.handle_xaxis",
    "observation.hinge_direction",
]

AUGMENTED_FEATURE_NAME_BY_KEY = {
    "observation.handle_pos": "handle_pos",
    "observation.handle_to_eef_pos": "handle_to_eef_pos",
    "observation.door_openness": "door_openness",
    "observation.handle_xaxis": "handle_xaxis",
    "observation.hinge_direction": "hinge_direction",
}


def flatten_value(value) -> Optional[np.ndarray]:
    """Convert a scalar / list / ndarray value to a flat float32 vector."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
        return arr.reshape(-1)
    if isinstance(value, (bool, int, float, np.number)):
        return np.asarray([float(value)], dtype=np.float32)
    return None


def pad_or_trim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad with zeros or trim to exactly target_dim."""
    if vec.shape[0] == target_dim:
        return vec.astype(np.float32, copy=False)
    if vec.shape[0] < target_dim:
        return np.pad(vec, (0, target_dim - vec.shape[0])).astype(np.float32, copy=False)
    return vec[:target_dim].astype(np.float32, copy=False)


def flatten_lowdim_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all non-image ndarray observations in sorted-key order."""
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if key.endswith("_image"):
            continue
        if isinstance(val, np.ndarray):
            parts.append(val.reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def remap_dataset_action_to_env(action: np.ndarray) -> np.ndarray:
    """
    Convert LeRobot/OpenCabinet dataset action layout to robosuite env layout.

    Dataset (len=12):
      [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Env (len=12):
      [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)]

    If action dim is not 12+, this is a no-op.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] < 12:
        return a.astype(np.float32, copy=False)

    out = a.copy()
    out[0:3] = a[5:8]     # eef_pos
    out[3:6] = a[8:11]    # eef_rot
    out[6] = a[11]        # gripper
    out[7:10] = a[0:3]    # base motion
    out[10] = a[3]        # torso
    out[11] = a[4]        # base mode / control mode
    if out.shape[0] > 12:
        out[12:] = a[12:]
    return out.astype(np.float32, copy=False)


def binarize_discrete_action_dims(
    action: np.ndarray,
    gripper_threshold: float = 0.0,
    base_mode_threshold: float = 0.0,
) -> np.ndarray:
    """
    Binarize signed discrete controls using midpoint threshold 0.0.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if a.shape[0] > 6:
        a[6] = 1.0 if a[6] >= float(gripper_threshold) else -1.0
    if a.shape[0] > 11:
        a[11] = 1.0 if a[11] >= float(base_mode_threshold) else -1.0
    return a.astype(np.float32, copy=False)


def postprocess_policy_action(action: np.ndarray, env_action_dim: int) -> np.ndarray:
    """
    Post-process policy action before env.step():
    1) dataset->env action order remap (when dim>=12)
    2) binarize gripper/base-mode at threshold 0.0
    3) pad/trim to env_action_dim
    """
    a = np.nan_to_num(
        np.asarray(action, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    a = remap_dataset_action_to_env(a)
    a = binarize_discrete_action_dims(a, gripper_threshold=0.0, base_mode_threshold=0.0)
    return pad_or_trim(a, env_action_dim)


def _safe_torch_load(path: str, device: torch.device):
    """Load checkpoints across PyTorch versions that may not support weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _unwrap_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize checkpoint state dict keys.

    Older diffusion checkpoints may save DiffusionPolicyCore.state_dict(), which
    prefixes all U-Net keys with "model.". This helper strips that prefix when
    present so we can load into ConditionalUnet1D directly.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    if any(k.startswith("model.") for k in state_dict.keys()):
        stripped = {
            k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
        }
        if stripped:
            return stripped
    return state_dict


def _sort_numeric_suffix(cols: Iterable[str], prefix: str) -> List[str]:
    def key_fn(name: str):
        suffix = name[len(prefix) :]
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(cols, key=key_fn)


def _find_data_root(dataset_path: Path) -> Path:
    """Resolve the base path containing LeRobot data."""
    if (dataset_path / "data").exists():
        return dataset_path
    if (dataset_path / "lerobot" / "data").exists():
        return dataset_path / "lerobot"
    raise FileNotFoundError(
        f"Could not find dataset data directory under {dataset_path}. "
        "Expected either data/chunk-000 or lerobot/data/chunk-000"
    )


def discover_parquet_files(
    dataset_path: str,
    prefer_augmented: bool = True,
) -> Tuple[List[Path], str]:
    """Discover parquet files, preferring 05b-augmented files when available."""
    base = Path(dataset_path)

    if prefer_augmented:
        aug_dir = base / "augmented"
        if aug_dir.exists():
            aug_files = sorted(aug_dir.glob("*.parquet"))
            if aug_files:
                return aug_files, "augmented"

    data_root = _find_data_root(base)
    chunk_dir = data_root / "data" / "chunk-000"
    files = sorted(chunk_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {chunk_dir}")
    return files, "raw"


def infer_state_columns(
    column_names: Sequence[str],
    include_augmented: bool = True,
    include_all_lowdim: bool = False,
    use_observation_state: bool = True,
) -> List[str]:
    """Infer which parquet columns should form the low-dimensional policy state."""
    cols = list(column_names)
    state_cols: List[str] = []

    use_packed_state = bool(use_observation_state and "observation.state" in cols)
    if use_packed_state:
        state_cols.append("observation.state")

    if include_augmented:
        for k in AUGMENTED_STATE_KEYS:
            if k in cols:
                state_cols.append(k)

    excluded = set(state_cols)
    if not use_observation_state:
        excluded.add("observation.state")

    if include_all_lowdim:
        lowdim_cols = [
            c
            for c in cols
            if c.startswith("observation.") and "image" not in c and c not in excluded
        ]
        state_cols.extend(sorted(lowdim_cols))
    elif not use_packed_state:
        # If no packed state column exists, include all low-dim observation keys.
        lowdim_cols = sorted(
            c
            for c in cols
            if c.startswith("observation.") and "image" not in c and c not in excluded
        )
        state_cols = lowdim_cols + [c for c in state_cols if c not in lowdim_cols]

    if not state_cols:
        # Fallback when observation.state is absent: use all non-image observation keys.
        state_cols = sorted(
            c
            for c in cols
            if c.startswith("observation.")
            and "image" not in c
            and (use_observation_state or c != "observation.state")
        )

    return state_cols


def infer_action_columns(column_names: Sequence[str]) -> List[str]:
    """Infer action columns from parquet schema."""
    cols = list(column_names)

    if "action" in cols:
        return ["action"]

    dotted = [c for c in cols if c.startswith("action.")]
    if dotted:
        return _sort_numeric_suffix(dotted, "action.")

    fallback = sorted(c for c in cols if "action" in c.lower())
    if fallback:
        return fallback

    raise ValueError("Could not infer action columns from parquet schema")


def series_to_2d(
    series,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Convert a pandas series to a 2D float32 array [N, D]."""
    values = series.to_numpy()
    rows: List[np.ndarray] = []
    dim = expected_dim

    for value in values:
        arr = flatten_value(value)
        if arr is None:
            if dim is None:
                dim = 1
            arr = np.zeros((dim,), dtype=np.float32)
        if dim is None:
            dim = int(arr.shape[0])
        if arr.shape[0] != dim:
            arr = pad_or_trim(arr, dim)
        rows.append(arr.astype(np.float32, copy=False))

    if dim is None:
        dim = 0
    if not rows:
        return np.zeros((0, dim), dtype=np.float32), dim

    return np.stack(rows, axis=0).astype(np.float32, copy=False), dim


class CabinetSequenceDataset(torch.utils.data.Dataset):
    """
    Sliding-window sequence dataset for low-dimensional diffusion policy training.

    Each sample provides:
      obs_seq:     [n_obs_steps, state_dim]
      action_seq:  [horizon, action_dim]
      phase:       int64 scalar  (0=approach, 1=grasp, 2=pull)

    Phase labels are derived automatically from handle distance and door openness:
      phase 0: dist_to_handle  > phase_approach_dist
      phase 1: dist_to_handle <= phase_approach_dist AND openness < phase_grasp_openness
      phase 2: door openness  >= phase_grasp_openness
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int,
        n_obs_steps: int,
        include_augmented: bool = True,
        include_all_lowdim: bool = False,
        use_observation_state: bool = True,
        state_keys_override: Optional[Sequence[str]] = None,
        max_episodes: Optional[int] = None,
        phase_approach_dist: float = 0.10,
        phase_grasp_openness: float = 0.05,
    ):
        super().__init__()

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for training. Install with: pip install pyarrow"
            ) from exc

        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.phase_approach_dist = float(phase_approach_dist)
        self.phase_grasp_openness = float(phase_grasp_openness)

        parquet_files, source = discover_parquet_files(
            dataset_path=dataset_path,
            prefer_augmented=include_augmented,
        )
        if max_episodes is not None:
            parquet_files = parquet_files[: max_episodes]

        if not parquet_files:
            raise RuntimeError("No parquet files available for dataset construction")

        first_table = pq.read_table(parquet_files[0])
        if state_keys_override:
            state_cols = [str(k) for k in state_keys_override]
            missing = [k for k in state_cols if k not in first_table.column_names]
            if missing:
                raise ValueError(
                    f"Requested state_keys_override not found in parquet schema: {missing}"
                )
        else:
            state_cols = infer_state_columns(
                first_table.column_names,
                include_augmented=include_augmented,
                include_all_lowdim=include_all_lowdim,
                use_observation_state=use_observation_state,
            )
        action_cols = infer_action_columns(first_table.column_names)

        self.state_keys = list(state_cols)
        self.action_keys = list(action_cols)
        self.state_key_dims: Dict[str, int] = {}
        self.action_key_dims: Dict[str, int] = {}
        self.source = source

        self.episodes_states: List[np.ndarray] = []
        self.episodes_actions: List[np.ndarray] = []
        self.episodes_phases: List[np.ndarray] = []

        # Phase feature columns — read from parquet even if not in state_cols.
        _phase_cols = ["observation.handle_to_eef_pos", "observation.door_openness"]
        _all_col_names = set(first_table.column_names)
        extra_phase_cols = [
            c for c in _phase_cols
            if c in _all_col_names and c not in state_cols and c not in action_cols
        ]

        for pf in parquet_files:
            read_cols = list(set(state_cols + action_cols + extra_phase_cols))
            table = pq.read_table(pf, columns=read_cols)
            df = table.to_pandas()
            if len(df) == 0:
                continue

            state_blocks = []
            for key in state_cols:
                arr, dim = series_to_2d(df[key], self.state_key_dims.get(key))
                self.state_key_dims[key] = dim
                state_blocks.append(arr)

            action_blocks = []
            for key in action_cols:
                arr, dim = series_to_2d(df[key], self.action_key_dims.get(key))
                self.action_key_dims[key] = dim
                action_blocks.append(arr)

            state_arr = np.concatenate(state_blocks, axis=1).astype(np.float32, copy=False)
            action_arr = np.concatenate(action_blocks, axis=1).astype(np.float32, copy=False)

            if len(state_arr) != len(action_arr):
                n = min(len(state_arr), len(action_arr))
                state_arr = state_arr[:n]
                action_arr = action_arr[:n]

            # Need enough context and prediction horizon for at least one training window.
            min_len = max(self.horizon, self.n_obs_steps)
            if len(state_arr) < min_len:
                continue

            # --- Phase labeling ---
            n = len(state_arr)
            if "observation.handle_to_eef_pos" in df.columns:
                h2e, _ = series_to_2d(df["observation.handle_to_eef_pos"])
                dists = np.linalg.norm(h2e[:n, :min(3, h2e.shape[1])], axis=1)
            else:
                dists = np.full(n, float("inf"), dtype=np.float32)

            if "observation.door_openness" in df.columns:
                open_arr, _ = series_to_2d(df["observation.door_openness"])
                openness = open_arr[:n, 0]
            else:
                openness = np.zeros(n, dtype=np.float32)

            phases = np.zeros(n, dtype=np.int64)
            phases[
                (dists <= self.phase_approach_dist) & (openness < self.phase_grasp_openness)
            ] = 1
            phases[openness >= self.phase_grasp_openness] = 2

            self.episodes_states.append(state_arr)
            self.episodes_actions.append(action_arr)
            self.episodes_phases.append(phases)

        if not self.episodes_states:
            raise RuntimeError(
                "No usable episodes were loaded. Check dataset download and augmentation outputs."
            )

        self.state_dim = int(self.episodes_states[0].shape[1])
        self.action_dim = int(self.episodes_actions[0].shape[1])

        for ep in self.episodes_states:
            if ep.shape[1] != self.state_dim:
                raise RuntimeError("Inconsistent state dimensions across episodes")
        for ep in self.episodes_actions:
            if ep.shape[1] != self.action_dim:
                raise RuntimeError("Inconsistent action dimensions across episodes")

        self.sample_index: List[Tuple[int, int]] = []
        for ep_idx, action_ep in enumerate(self.episodes_actions):
            t_min = self.n_obs_steps - 1
            t_max = len(action_ep) - self.horizon
            for t in range(t_min, t_max + 1):
                self.sample_index.append((ep_idx, t))

        if not self.sample_index:
            raise RuntimeError("No sliding windows could be generated from loaded episodes")

        # Normalization stats over all timesteps.
        all_states = np.concatenate(self.episodes_states, axis=0)
        all_actions = np.concatenate(self.episodes_actions, axis=0)

        self.state_mean = all_states.mean(axis=0).astype(np.float32)
        self.state_std = all_states.std(axis=0).astype(np.float32)
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32)

        self.state_std = np.clip(self.state_std, 1e-3, None)
        self.action_std = np.clip(self.action_std, 1e-3, None)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        ep_idx, t = self.sample_index[idx]
        states = self.episodes_states[ep_idx]
        actions = self.episodes_actions[ep_idx]
        phases = self.episodes_phases[ep_idx]

        obs_seq = states[t - self.n_obs_steps + 1 : t + 1]
        action_seq = actions[t : t + self.horizon]
        phase = int(phases[t])

        return (
            torch.from_numpy(obs_seq.astype(np.float32, copy=False)),
            torch.from_numpy(action_seq.astype(np.float32, copy=False)),
            torch.tensor(phase, dtype=torch.long),
        )


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = _make_group_norm(out_channels)
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_channels * 2))
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        scale_shift = self.cond_proj(cond).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        h = h * (1.0 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ConditionalUnet1D(nn.Module):
    """Compact 1D Conv U-Net for low-dimensional diffusion policy."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        n_obs_steps: int,
        horizon: int,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2),
        num_res_blocks: int = 1,
        time_emb_dim: int = 128,
        cond_dim: int = 256,
        dropout: float = 0.0,
        num_phases: int = 1,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.num_phases = int(num_phases)

        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.obs_mlp = nn.Sequential(
            nn.Linear(self.state_dim * self.n_obs_steps, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Phase embedding: learned vector per stage added to the conditioning.
        # Only created when num_phases > 1 so old checkpoints load unchanged.
        if self.num_phases > 1:
            self.phase_emb = nn.Embedding(self.num_phases, cond_dim)

        self.init_conv = nn.Conv1d(self.action_dim, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        level_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * int(mult)
            resblocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                resblocks.append(
                    ConditionalResBlock1D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        cond_dim=cond_dim,
                        dropout=dropout,
                    )
                )
                in_ch = out_ch

            downsample = Downsample1D(in_ch) if level < len(channel_mults) - 1 else None
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "resblocks": resblocks,
                        "downsample": downsample if downsample is not None else nn.Identity(),
                    }
                )
            )
            level_channels.append(out_ch)

        self.mid_block1 = ConditionalResBlock1D(in_ch, in_ch, cond_dim=cond_dim, dropout=dropout)
        self.mid_block2 = ConditionalResBlock1D(in_ch, in_ch, cond_dim=cond_dim, dropout=dropout)

        self.up_blocks = nn.ModuleList()
        for rev_level, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * int(mult)
            skip_ch = level_channels[rev_level]

            resblocks = nn.ModuleList()
            resblocks.append(
                ConditionalResBlock1D(
                    in_channels=in_ch + skip_ch,
                    out_channels=out_ch,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
            )
            for _ in range(num_res_blocks - 1):
                resblocks.append(
                    ConditionalResBlock1D(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        cond_dim=cond_dim,
                        dropout=dropout,
                    )
                )

            upsample = Upsample1D(out_ch) if rev_level > 0 else None
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "resblocks": resblocks,
                        "upsample": upsample if upsample is not None else nn.Identity(),
                    }
                )
            )
            in_ch = out_ch

        self.final_norm = _make_group_norm(in_ch)
        self.final_conv = nn.Conv1d(in_ch, self.action_dim, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        obs_seq: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: [B, horizon, action_dim]
            timesteps:    [B]
            obs_seq:      [B, n_obs_steps, state_dim]
            phase:        [B] long, optional — 0=approach, 1=grasp, 2=pull
        Returns:
            predicted_noise: [B, horizon, action_dim]
        """
        cond_t = self.time_mlp(self.time_emb(timesteps))
        cond_obs = self.obs_mlp(obs_seq.reshape(obs_seq.shape[0], -1))
        cond = cond_t + cond_obs
        if phase is not None and hasattr(self, "phase_emb"):
            cond = cond + self.phase_emb(phase.long())

        x = noisy_action.transpose(1, 2)
        x = self.init_conv(x)

        skips = []
        for block in self.down_blocks:
            for res in block["resblocks"]:
                x = res(x, cond)
            skips.append(x)
            x = block["downsample"](x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for block in self.up_blocks:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            for res in block["resblocks"]:
                x = res(x, cond)
            x = block["upsample"](x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x.transpose(1, 2)


class DDPMScheduler:
    """Minimal DDPM scheduler for low-dimensional action diffusion."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.num_train_timesteps = int(num_train_timesteps)

        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def _extract(self, vec: torch.Tensor, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        out = vec.to(device=t.device)[t]
        while out.dim() < len(target_shape):
            out = out.unsqueeze(-1)
        return out

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_a = self._extract(self.sqrt_alphas_cumprod, t, clean.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alphas_cumprod, t, clean.shape)
        return sqrt_a * clean + sqrt_om * noise

    def step(
        self,
        pred_noise: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        bsz = sample.shape[0]
        t = torch.full((bsz,), int(timestep), device=sample.device, dtype=torch.long)

        alpha_t = self._extract(self.alphas, t, sample.shape)
        beta_t = self._extract(self.betas, t, sample.shape)
        acp_t = self._extract(self.alphas_cumprod, t, sample.shape)
        acp_prev = self._extract(self.alphas_cumprod_prev, t, sample.shape)

        # Predict x0 from epsilon parameterization.
        pred_x0 = (sample - torch.sqrt(1.0 - acp_t) * pred_noise) / torch.sqrt(acp_t)

        coef1 = beta_t * torch.sqrt(acp_prev) / (1.0 - acp_t)
        coef2 = (1.0 - acp_prev) * torch.sqrt(alpha_t) / (1.0 - acp_t)
        mean = coef1 * pred_x0 + coef2 * sample

        var = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
        var = torch.clamp(var, min=1e-20)

        if deterministic:
            noise = torch.zeros_like(sample)
        else:
            noise = torch.randn_like(sample)

        nonzero_mask = (t > 0).float()
        while nonzero_mask.dim() < sample.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)

        return mean + nonzero_mask * torch.sqrt(var) * noise


class DiffusionPolicyCore(nn.Module):
    """Thin wrapper that pairs the U-Net with scheduler sampling."""

    def __init__(self, model: ConditionalUnet1D, scheduler: DDPMScheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    @property
    def action_dim(self) -> int:
        return self.model.action_dim

    @property
    def horizon(self) -> int:
        return self.model.horizon

    def forward(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        obs_seq: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
    ):
        return self.model(noisy_action, timesteps, obs_seq, phase=phase)

    @torch.no_grad()
    def sample(
        self,
        obs_seq: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        bsz = obs_seq.shape[0]
        device = obs_seq.device

        sample = torch.randn(
            (bsz, self.horizon, self.action_dim),
            device=device,
            dtype=obs_seq.dtype,
        )

        n_train = self.scheduler.num_train_timesteps
        if num_inference_steps is None or num_inference_steps >= n_train:
            timesteps = list(range(n_train - 1, -1, -1))
        else:
            t = np.linspace(n_train - 1, 0, num_inference_steps, dtype=np.int64)
            timesteps = [int(x) for x in t.tolist()]

        for t in timesteps:
            t_batch = torch.full((bsz,), t, device=device, dtype=torch.long)
            pred_noise = self.model(sample, t_batch, obs_seq, phase=phase)
            sample = self.scheduler.step(
                pred_noise=pred_noise,
                timestep=t,
                sample=sample,
                deterministic=deterministic,
            )

        return sample


class SimplePolicyNet(nn.Module):
    """Legacy MLP used by earlier checkpoints."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class StageConditionedBCNet(nn.Module):
    """Phase-conditioned BC network with separate heads for approach / grasp / pull."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_phases: int = 3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_phases = int(num_phases)

        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, self.action_dim), nn.Tanh()) for _ in range(self.num_phases)]
        )

    def forward(self, state: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x = self.trunk(state)
        phase = phase.long().view(-1)
        outputs = torch.stack([head(x) for head in self.heads], dim=1)  # [B, P, A]
        idx = phase.view(-1, 1, 1).expand(-1, 1, self.action_dim)
        return outputs.gather(1, idx).squeeze(1)


# -----------------------------------------------------------------------------
# Online handle features (05b parity at inference)
# -----------------------------------------------------------------------------

def find_fixture_handle_bodies(model, fixture_name: str) -> List[str]:
    handles = []
    for name in model.body_names:
        if not name:
            continue
        if fixture_name in name and "handle" in name:
            handles.append(name)
    return handles


def find_fixture_door_joints(model, fixture_name: str) -> List[Tuple[str, int]]:
    joints = []
    for name in model.joint_names:
        if not name:
            continue
        if fixture_name in name and "door" in name:
            joints.append((name, model.joint_name2id(name)))
    return joints


def compute_door_openness(model, data, door_joints: Sequence[Tuple[str, int]]) -> float:
    if not door_joints:
        return 0.0

    openness_vals = []
    for jname, jidx in door_joints:
        addr = model.get_joint_qpos_addr(jname)
        if isinstance(addr, tuple):
            addr = addr[0]
        qpos = float(data.qpos[addr])
        jmin, jmax = model.jnt_range[jidx]
        if jmax - jmin <= 1e-8:
            openness = 0.0
        else:
            # Closed bound is whichever bound is closer to zero.
            if abs(jmin) < abs(jmax):
                openness = abs(qpos - jmin) / (jmax - jmin)
            else:
                openness = abs(qpos - jmax) / (jmax - jmin)
        openness_vals.append(float(np.clip(openness, 0.0, 1.0)))

    return float(np.mean(openness_vals))


def build_handle_to_joint_map(
    handle_bodies: Sequence[str],
    door_joints: Sequence[Tuple[str, int]],
) -> Dict[str, List[Tuple[str, int]]]:
    if len(handle_bodies) <= 1 or len(door_joints) <= 1:
        return {hb: list(door_joints) for hb in handle_bodies}

    mapping = {}
    for hb in handle_bodies:
        hb_lower = hb.lower()
        if "left" in hb_lower:
            matched = [(jn, ji) for (jn, ji) in door_joints if "left" in jn.lower()]
        elif "right" in hb_lower:
            matched = [(jn, ji) for (jn, ji) in door_joints if "right" in jn.lower()]
        else:
            matched = []
        mapping[hb] = matched if matched else list(door_joints)
    return mapping


def hinge_direction_for_handle(
    handle_body: str,
    handle_to_joint_map: Dict[str, List[Tuple[str, int]]],
    model,
) -> float:
    joints = handle_to_joint_map.get(handle_body, [])
    if not joints:
        return 0.0
    _, jidx = joints[0]
    jmin, jmax = model.jnt_range[jidx]
    return 1.0 if abs(jmin) < abs(jmax) else -1.0


class OnlineHandleFeatureExtractor:
    """
    Computes the same handle-centric features added by 05b for live evaluation.
    """

    def __init__(self, open_threshold: float = 0.90):
        self.open_threshold = float(open_threshold)
        self._cache_key: Optional[Tuple[int, str]] = None
        self._cache = {}

    def _ensure_cache(self, env):
        fixture_name = getattr(getattr(env, "fxtr", None), "name", "")
        model = env.sim.model
        key = (id(model), fixture_name)
        if key == self._cache_key:
            return

        handle_bodies = find_fixture_handle_bodies(model, fixture_name)
        door_joints = find_fixture_door_joints(model, fixture_name)
        handle_to_joint_map = build_handle_to_joint_map(handle_bodies, door_joints)

        self._cache = {
            "fixture_name": fixture_name,
            "handle_bodies": handle_bodies,
            "door_joints": door_joints,
            "handle_to_joint_map": handle_to_joint_map,
            "handle_body_ids": {name: model.body_name2id(name) for name in handle_bodies},
        }
        self._cache_key = key

    def _get_eef_pos(self, env) -> np.ndarray:
        robot = env.robots[0]

        eef_site_id = getattr(robot, "eef_site_id", None)
        if isinstance(eef_site_id, dict) and eef_site_id:
            arm = "right" if "right" in eef_site_id else next(iter(eef_site_id.keys()))
            sid = eef_site_id[arm]
            return np.array(env.sim.data.site_xpos[sid], dtype=np.float32)
        if isinstance(eef_site_id, (int, np.integer)):
            return np.array(env.sim.data.site_xpos[int(eef_site_id)], dtype=np.float32)

        # Fallback to eef body name if site id is unavailable.
        eef_name = getattr(robot.robot_model, "eef_name", None)
        if isinstance(eef_name, dict) and eef_name:
            arm = "right" if "right" in eef_name else next(iter(eef_name.keys()))
            body_name = eef_name[arm]
            if body_name in env.sim.model.body_names:
                return np.array(env.sim.data.get_body_xpos(body_name), dtype=np.float32)
        elif isinstance(eef_name, str) and eef_name in env.sim.model.body_names:
            return np.array(env.sim.data.get_body_xpos(eef_name), dtype=np.float32)

        return np.zeros((3,), dtype=np.float32)

    def extract(self, env) -> Dict[str, np.ndarray]:
        self._ensure_cache(env)

        model = env.sim.model
        data = env.sim.data

        handle_bodies: List[str] = self._cache.get("handle_bodies", [])
        handle_to_joint_map = self._cache.get("handle_to_joint_map", {})
        body_ids = self._cache.get("handle_body_ids", {})

        if not handle_bodies:
            return {
                "handle_pos": np.zeros((3,), dtype=np.float32),
                "handle_to_eef_pos": np.zeros((3,), dtype=np.float32),
                "door_openness": np.zeros((1,), dtype=np.float32),
                "handle_xaxis": np.zeros((3,), dtype=np.float32),
                "hinge_direction": np.zeros((1,), dtype=np.float32),
            }

        eef_pos = self._get_eef_pos(env)

        per_door = {
            hb: compute_door_openness(model, data, handle_to_joint_map.get(hb, []))
            for hb in handle_bodies
        }

        active = [hb for hb in handle_bodies if per_door[hb] < self.open_threshold]
        candidates = active if active else handle_bodies

        dists = []
        for hb in candidates:
            bid = body_ids[hb]
            dists.append(np.linalg.norm(data.body_xpos[bid] - eef_pos))
        target = candidates[int(np.argmin(dists))]
        target_id = body_ids[target]

        handle_pos = np.array(data.body_xpos[target_id], dtype=np.float32)
        handle_to_eef = handle_pos - eef_pos
        openness = np.array([per_door[target]], dtype=np.float32)

        xmat = np.array(data.body_xmat[target_id], dtype=np.float32).reshape(3, 3)
        handle_xaxis = xmat[:, 0].astype(np.float32)

        hinge_dir = np.array(
            [hinge_direction_for_handle(target, handle_to_joint_map, model)],
            dtype=np.float32,
        )

        return {
            "handle_pos": handle_pos,
            "handle_to_eef_pos": handle_to_eef,
            "door_openness": openness,
            "handle_xaxis": handle_xaxis,
            "hinge_direction": hinge_dir,
        }


def build_state_vector(
    obs: Dict[str, np.ndarray],
    state_keys: Optional[Sequence[str]],
    state_key_dims: Optional[Dict[str, int]],
    state_dim: int,
    env=None,
    handle_extractor: Optional[OnlineHandleFeatureExtractor] = None,
) -> np.ndarray:
    """Build state vector in the exact feature order used during training."""

    if not state_keys:
        state = flatten_lowdim_obs(obs)
        return pad_or_trim(state, state_dim)

    lowdim_state_cache = None
    handle_features = None
    parts = []

    for key in state_keys:
        expected_dim = None
        if state_key_dims is not None and key in state_key_dims:
            expected_dim = int(state_key_dims[key])

        stripped = key[len("observation.") :] if key.startswith("observation.") else key
        value = None

        if key == "observation.state" or stripped == "state":
            if "state" in obs:
                value = obs.get("state")
            elif "observation.state" in obs:
                value = obs.get("observation.state")
            else:
                if lowdim_state_cache is None:
                    lowdim_state_cache = flatten_lowdim_obs(obs)
                value = lowdim_state_cache
        elif stripped in obs:
            value = obs[stripped]
        elif key in obs:
            value = obs[key]
        elif key in AUGMENTED_FEATURE_NAME_BY_KEY:
            if handle_features is None and env is not None and handle_extractor is not None:
                handle_features = handle_extractor.extract(env)
            feat_key = AUGMENTED_FEATURE_NAME_BY_KEY[key]
            if handle_features is not None:
                value = handle_features.get(feat_key, None)
        elif stripped in AUGMENTED_FEATURE_NAME_BY_KEY.values():
            if handle_features is None and env is not None and handle_extractor is not None:
                handle_features = handle_extractor.extract(env)
            if handle_features is not None:
                value = handle_features.get(stripped, None)

        arr = flatten_value(value)
        if arr is None:
            arr = np.zeros((expected_dim or 0,), dtype=np.float32)
        if expected_dim is not None:
            arr = pad_or_trim(arr, expected_dim)
        parts.append(arr.astype(np.float32, copy=False))

    state = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros((0,), dtype=np.float32)
    return pad_or_trim(state, state_dim)


def is_one_door_open_success(env, threshold: float = 0.10) -> bool:
    """
    Success when any door joint on the target fixture is open enough.

    Falls back to env._check_success() if fixture state is unavailable.
    """
    fxtr = getattr(env, "fxtr", None)
    if fxtr is not None:
        if hasattr(fxtr, "get_door_state"):
            try:
                door_state = fxtr.get_door_state(env=env)
                if isinstance(door_state, dict) and door_state:
                    vals = [float(v) for v in door_state.values()]
                    if vals:
                        return bool(any(v >= threshold for v in vals))
            except Exception:
                pass

        if hasattr(fxtr, "door_joint_names") and hasattr(fxtr, "get_joint_state"):
            try:
                joint_names = list(fxtr.door_joint_names)
                if joint_names:
                    joint_state = fxtr.get_joint_state(env, joint_names)
                    vals = [float(joint_state[j]) for j in joint_names if j in joint_state]
                    if vals:
                        return bool(any(v >= threshold for v in vals))
            except Exception:
                pass

    return bool(env._check_success())


# -----------------------------------------------------------------------------
# Policy wrappers for evaluation / rollout scripts
# -----------------------------------------------------------------------------

@dataclass
class PolicyInfo:
    model_type: str
    state_dim: int
    action_dim: int
    epoch: int
    loss: float


class BasePolicyWrapper:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_keys: Optional[Sequence[str]] = None,
        state_key_dims: Optional[Dict[str, int]] = None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state_keys = list(state_keys) if state_keys is not None else None
        self.state_key_dims = dict(state_key_dims) if state_key_dims is not None else None

    def reset(self):
        raise NotImplementedError

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        raise NotImplementedError


def _find_state_slice(
    state_keys: Optional[Sequence[str]],
    state_key_dims: Optional[Dict[str, int]],
    target_keys: Sequence[str],
) -> Optional[slice]:
    if not state_keys or not state_key_dims:
        return None

    targets = set(target_keys)
    offset = 0
    for key in state_keys:
        dim = int(state_key_dims.get(key, 0))
        stripped = key[len("observation.") :] if key.startswith("observation.") else key
        if key in targets or stripped in targets:
            return slice(offset, offset + dim)
        offset += dim
    return None


class SimplePolicyWrapper(BasePolicyWrapper):
    def __init__(
        self,
        model: SimplePolicyNet,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        state_keys: Optional[Sequence[str]] = None,
        state_key_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__(state_dim, action_dim, state_keys, state_key_dims)
        self.model = model
        self.device = device
        self.handle_extractor = OnlineHandleFeatureExtractor()

    def reset(self):
        return

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        state = build_state_vector(
            obs=obs,
            state_keys=self.state_keys,
            state_key_dims=self.state_key_dims,
            state_dim=self.state_dim,
            env=env,
            handle_extractor=self.handle_extractor,
        )
        with torch.no_grad():
            action = self.model(torch.from_numpy(state).unsqueeze(0).to(self.device))
            action = action.cpu().numpy().squeeze(0)
        return postprocess_policy_action(action, env_action_dim)


class StagedBCPolicyWrapper(BasePolicyWrapper):
    """
    Runtime controller with explicit phases:
    0 = approach handle, 1 = close gripper, 2 = pull.
    """

    def __init__(
        self,
        model: StageConditionedBCNet,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        state_keys: Sequence[str],
        state_key_dims: Dict[str, int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        approach_dist_threshold: float,
        reapproach_dist_threshold: float,
        grasp_steps: int,
        gripper_close_value: float,
    ):
        super().__init__(state_dim, action_dim, state_keys, state_key_dims)
        self.model = model
        self.device = device
        self.handle_extractor = OnlineHandleFeatureExtractor()

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = np.clip(state_std.astype(np.float32), 1e-3, None)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = np.clip(action_std.astype(np.float32), 1e-3, None)

        self.approach_dist_threshold = float(approach_dist_threshold)
        self.reapproach_dist_threshold = float(reapproach_dist_threshold)
        self.grasp_steps = int(max(grasp_steps, 1))
        self.gripper_close_value = float(gripper_close_value)

        self.handle_to_eef_slice = _find_state_slice(
            self.state_keys,
            self.state_key_dims,
            ["observation.handle_to_eef_pos", "handle_to_eef_pos"],
        )
        self.door_openness_slice = _find_state_slice(
            self.state_keys,
            self.state_key_dims,
            ["observation.door_openness", "door_openness"],
        )

        self.phase = 0
        self.grasp_countdown = 0
        self.prev_action: Optional[np.ndarray] = None

    def reset(self):
        self.phase = 0
        self.grasp_countdown = 0
        self.prev_action = None

    def _get_action_bounds(self, env, env_action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        low = np.full((env_action_dim,), -1.0, dtype=np.float32)
        high = np.full((env_action_dim,), 1.0, dtype=np.float32)

        spec = getattr(env, "action_spec", None)
        if spec is not None:
            try:
                spec_low, spec_high = spec
                low_arr = flatten_value(spec_low)
                high_arr = flatten_value(spec_high)
                if low_arr is not None and high_arr is not None:
                    low = pad_or_trim(low_arr.astype(np.float32, copy=False), env_action_dim)
                    high = pad_or_trim(high_arr.astype(np.float32, copy=False), env_action_dim)
                    lo = np.minimum(low, high)
                    hi = np.maximum(low, high)
                    low, high = lo, hi
            except Exception:
                pass

        return low, high

    def _extract_dist_and_openness(self, state: np.ndarray) -> Tuple[float, float, Optional[np.ndarray]]:
        dist = float("inf")
        handle_to_eef = None
        if self.handle_to_eef_slice is not None:
            h = state[self.handle_to_eef_slice]
            if h.shape[0] >= 3:
                handle_to_eef = h[:3].astype(np.float32, copy=False)
                dist = float(np.linalg.norm(handle_to_eef))

        openness = 0.0
        if self.door_openness_slice is not None:
            o = state[self.door_openness_slice]
            if o.shape[0] > 0:
                openness = float(o[0])

        return dist, openness, handle_to_eef

    def _update_phase(self, dist: float):
        if np.isfinite(dist) and dist > self.reapproach_dist_threshold:
            self.phase = 0
            self.grasp_countdown = 0
            return

        if self.phase == 0:
            if np.isfinite(dist) and dist <= self.approach_dist_threshold:
                self.phase = 1
                self.grasp_countdown = self.grasp_steps
            return

        if self.phase == 1:
            self.grasp_countdown -= 1
            if self.grasp_countdown <= 0:
                self.phase = 2

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        state = build_state_vector(
            obs=obs,
            state_keys=self.state_keys,
            state_key_dims=self.state_key_dims,
            state_dim=self.state_dim,
            env=env,
            handle_extractor=self.handle_extractor,
        )
        dist, openness, handle_to_eef = self._extract_dist_and_openness(state)
        self._update_phase(dist)

        state_norm = (state - self.state_mean) / self.state_std
        s = torch.from_numpy(state_norm).unsqueeze(0).to(self.device)
        p = torch.tensor([self.phase], dtype=torch.long, device=self.device)
        with torch.no_grad():
            action_norm = self.model(s, p).cpu().numpy().squeeze(0)
        action = action_norm * self.action_std + self.action_mean
        action = postprocess_policy_action(action, env_action_dim)

        # Mild geometric attractor during approach for more direct handle targeting.
        if self.phase == 0 and handle_to_eef is not None and action.shape[0] >= 3:
            attract = np.clip(0.8 * handle_to_eef, -0.10, 0.10)
            action[:3] = 0.65 * action[:3] + 0.35 * attract

        # During grasp phase, force persistent close command on the gripper axis.
        if self.phase == 1 and action.shape[0] >= 7:
            action[6] = self.gripper_close_value

        if self.prev_action is not None and self.prev_action.shape == action.shape:
            delta = action - self.prev_action
            if action.shape[0] >= 3:
                delta[:3] = np.clip(delta[:3], -0.08, 0.08)
            if action.shape[0] >= 6:
                delta[3:6] = np.clip(delta[3:6], -0.18, 0.18)
            if action.shape[0] >= 7:
                delta[6:] = np.clip(delta[6:], -0.35, 0.35)
            action = self.prev_action + delta

        low, high = self._get_action_bounds(env, env_action_dim)
        action = np.clip(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0), low, high)
        action = binarize_discrete_action_dims(action, gripper_threshold=0.0, base_mode_threshold=0.0)
        self.prev_action = action.astype(np.float32, copy=True)
        return self.prev_action


class TemporalUnetBCPolicyWrapper(BasePolicyWrapper):
    """
    Non-staged temporal behavior cloning wrapper using the 1D U-Net directly.

    The model predicts an action chunk from recent state history, and we execute
    the first n_action_steps in closed loop.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        state_keys: Sequence[str],
        state_key_dims: Dict[str, int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        n_obs_steps: int,
        n_action_steps: int,
    ):
        super().__init__(state_dim, action_dim, state_keys, state_key_dims)
        self.model = model
        self.device = device

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = np.clip(state_std.astype(np.float32), 1e-3, None)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = np.clip(action_std.astype(np.float32), 1e-3, None)

        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.horizon = int(model.horizon)

        self.obs_history: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self.action_queue: deque[np.ndarray] = deque()
        self.handle_extractor = OnlineHandleFeatureExtractor()

    def reset(self):
        self.obs_history.clear()
        self.action_queue.clear()

    def _get_obs_context(self) -> np.ndarray:
        if not self.obs_history:
            return np.zeros((self.n_obs_steps, self.state_dim), dtype=np.float32)

        seq = list(self.obs_history)
        if len(seq) < self.n_obs_steps:
            seq = [seq[0]] * (self.n_obs_steps - len(seq)) + seq
        return np.stack(seq, axis=0).astype(np.float32)

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        state = build_state_vector(
            obs=obs,
            state_keys=self.state_keys,
            state_key_dims=self.state_key_dims,
            state_dim=self.state_dim,
            env=env,
            handle_extractor=self.handle_extractor,
        )
        self.obs_history.append(state)

        if not self.action_queue:
            obs_seq = self._get_obs_context()
            obs_norm = (obs_seq - self.state_mean) / self.state_std
            obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zero_actions = torch.zeros(
                    (1, self.horizon, self.action_dim),
                    device=self.device,
                    dtype=obs_tensor.dtype,
                )
                zero_t = torch.zeros((1,), device=self.device, dtype=torch.long)
                pred_action_norm = self.model(
                    noisy_action=zero_actions,
                    timesteps=zero_t,
                    obs_seq=obs_tensor,
                    phase=None,
                )

            pred_action_norm = pred_action_norm.squeeze(0).cpu().numpy()
            pred_action = pred_action_norm * self.action_std + self.action_mean
            for a in pred_action[: self.n_action_steps]:
                self.action_queue.append(postprocess_policy_action(a, env_action_dim))

        action = self.action_queue.popleft()
        return pad_or_trim(action, env_action_dim)


class DiffusionPolicyWrapper(BasePolicyWrapper):
    def __init__(
        self,
        policy: DiffusionPolicyCore,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        state_keys: Sequence[str],
        state_key_dims: Dict[str, int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        n_obs_steps: int,
        n_action_steps: int,
        num_inference_steps: int,
    ):
        super().__init__(state_dim, action_dim, state_keys, state_key_dims)
        self.policy = policy
        self.device = device

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = np.clip(state_std.astype(np.float32), 1e-3, None)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std = np.clip(action_std.astype(np.float32), 1e-3, None)

        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)

        self.obs_history: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self.action_queue: deque[np.ndarray] = deque()
        self.handle_extractor = OnlineHandleFeatureExtractor()
        # Kept for compatibility with optional staged wrappers.
        self.prev_action: Optional[np.ndarray] = None
        # Set by StagedDiffusionPolicyWrapper before each act() call so the
        # correct phase embedding is passed to policy.sample().
        self.current_phase: Optional[int] = None

    def reset(self):
        self.obs_history.clear()
        self.action_queue.clear()
        self.prev_action = None

    def _get_obs_context(self) -> np.ndarray:
        if not self.obs_history:
            return np.zeros((self.n_obs_steps, self.state_dim), dtype=np.float32)

        seq = list(self.obs_history)
        if len(seq) < self.n_obs_steps:
            pad = [seq[0]] * (self.n_obs_steps - len(seq))
            seq = pad + seq
        return np.stack(seq, axis=0).astype(np.float32)

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        state = build_state_vector(
            obs=obs,
            state_keys=self.state_keys,
            state_key_dims=self.state_key_dims,
            state_dim=self.state_dim,
            env=env,
            handle_extractor=self.handle_extractor,
        )
        self.obs_history.append(state)

        if not self.action_queue:
            obs_seq = self._get_obs_context()
            obs_norm = (obs_seq - self.state_mean) / self.state_std

            obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0).to(self.device)
            phase_tensor = None
            if self.current_phase is not None and self.policy.model.num_phases > 1:
                phase_tensor = torch.tensor(
                    [self.current_phase], dtype=torch.long, device=self.device
                )
            with torch.no_grad():
                pred_action_norm = self.policy.sample(
                    obs_seq=obs_tensor,
                    phase=phase_tensor,
                    num_inference_steps=self.num_inference_steps,
                    deterministic=True,
                )
            pred_action_norm = pred_action_norm.squeeze(0).cpu().numpy()
            pred_action = pred_action_norm * self.action_std + self.action_mean

            chunk = pred_action[: self.n_action_steps]
            for a in chunk:
                self.action_queue.append(postprocess_policy_action(a, env_action_dim))

        action = self.action_queue.popleft()
        return pad_or_trim(action, env_action_dim)


class StagedDiffusionPolicyWrapper(BasePolicyWrapper):
    """
    Wraps DiffusionPolicyWrapper with explicit three-phase stage control.

    Phase 0 - Approach: diffusion drives XYZ toward handle; rotations zeroed.
    Phase 1 - Grasp:    XYZ motion dampened to hold position near handle;
                        gripper forced closed for grasp_steps to lock grip.
    Phase 2 - Pull:     diffusion drives the full action (it learned pulling
                        from demonstrations and runs without override here).

    The phase transitions are purely geometry-driven (handle distance +
    countdown timer), so no retraining is required.
    """

    def __init__(
        self,
        inner: DiffusionPolicyWrapper,
        approach_dist_threshold: float = 0.05,
        reapproach_dist_threshold: float = 0.14,
        grasp_steps: int = 12,
        gripper_close_value: float = -1.0,
        rotation_freeze_dist: float = 0.10,
    ):
        super().__init__(
            inner.state_dim,
            inner.action_dim,
            inner.state_keys,
            inner.state_key_dims,
        )
        self.inner = inner
        self.approach_dist_threshold = float(approach_dist_threshold)
        self.reapproach_dist_threshold = float(reapproach_dist_threshold)
        self.grasp_steps = int(max(grasp_steps, 1))
        self.gripper_close_value = float(gripper_close_value)
        # Once the arm is within rotation_freeze_dist of the handle, orientation
        # is locked for the rest of the approach so the arm stops over-rotating.
        self.rotation_freeze_dist = float(rotation_freeze_dist)

        self.phase = 0
        self.grasp_countdown = 0
        self.rotation_frozen = False

    def reset(self):
        self.inner.reset()
        self.phase = 0
        self.grasp_countdown = 0
        self.rotation_frozen = False

    def _update_phase(self, dist: float):
        # If EEF drifts too far from handle, reset to approach and unfreeze
        # rotation so the arm can re-orient on the next approach.
        if np.isfinite(dist) and dist > self.reapproach_dist_threshold:
            if self.phase != 0:
                self.rotation_frozen = False
            self.phase = 0
            self.grasp_countdown = 0
            return

        if self.phase == 0:
            if np.isfinite(dist) and dist <= self.approach_dist_threshold:
                self.phase = 1
                self.grasp_countdown = self.grasp_steps
            return

        if self.phase == 1:
            self.grasp_countdown -= 1
            if self.grasp_countdown <= 0:
                self.phase = 2

    def act(self, obs: Dict[str, np.ndarray], env, env_action_dim: int) -> np.ndarray:
        # Measure distance to handle (extractor caches per model/fixture).
        handle_feat = self.inner.handle_extractor.extract(env)
        handle_to_eef = handle_feat.get("handle_to_eef_pos")

        dist = float("inf")
        if handle_to_eef is not None:
            h = np.asarray(handle_to_eef, dtype=np.float32).reshape(-1)
            if h.shape[0] >= 3:
                dist = float(np.linalg.norm(h[:3]))

        prev_phase = self.phase
        self._update_phase(dist)
        if self.phase != prev_phase:
            print(f"[StagedDiffusion] phase {prev_phase} -> {self.phase}  dist={dist:.3f}", flush=True)

        # Tell the inner wrapper which phase we're in so it passes the correct
        # phase embedding to policy.sample() (only used when model has num_phases > 1).
        self.inner.current_phase = self.phase

        # Diffusion model generates the base action (with its own stabilisation).
        action = self.inner.act(obs=obs, env=env, env_action_dim=env_action_dim)

        if self.phase == 0:
            # Rotation: allow dampened rotation while far so the arm can orient
            # the gripper correctly.  Once within rotation_freeze_dist, lock
            # orientation permanently for the rest of the approach — this stops
            # the arm from over-rotating past the target pose.
            if np.isfinite(dist) and dist <= self.rotation_freeze_dist and not self.rotation_frozen:
                self.rotation_frozen = True
                print(f"[StagedDiffusion] rotation FROZEN at dist={dist:.3f}", flush=True)

            if self.rotation_frozen and action.shape[0] >= 6:
                action[3:6] = 0.0
                # Keep inner delta-smoother consistent with what we actually execute.
                if self.inner.prev_action is not None and self.inner.prev_action.shape == action.shape:
                    self.inner.prev_action[3:6] = 0.0
            elif action.shape[0] >= 6:
                # Not yet frozen: slow rotation down further so the arm orients
                # gradually without overshooting.
                action[3:6] *= 0.25

            # XYZ: proportional geometric controller dominates when far, fades
            # out near the handle so the learned policy handles fine alignment.
            if (
                handle_to_eef is not None
                and action.shape[0] >= 3
                and np.isfinite(dist)
                and dist > 1e-3
            ):
                h = np.asarray(handle_to_eef, dtype=np.float32).reshape(-1)[:3]
                approach = np.clip(0.6 * h, -0.25, 0.25)
                # geo_weight: 0.80 when far (dist ≥ 0.22 m), fades to 0 at handle
                geo_weight = float(np.clip((dist - 0.07) / 0.15, 0.0, 0.80))
                action[:3] = geo_weight * approach + (1.0 - geo_weight) * action[:3]

        elif self.phase == 1:
            # Grasp: hold position (dampen XYZ), zero rotation, close gripper.
            # NOTE: we update inner.prev_action to the dampened value so its
            # delta smoother stays consistent with what we actually executed.
            if action.shape[0] >= 3:
                action[:3] *= 0.1
            if action.shape[0] >= 6:
                action[3:6] = 0.0
            if action.shape[0] >= 7:
                action[6] = self.gripper_close_value
            # Keep inner wrapper's prev_action in sync with actual executed action.
            self.inner.prev_action = action.astype(np.float32, copy=True)

        # Phase 2 (pull): diffusion action used as-is — the model learned to
        # pull the door open from the demonstrations.

        print(f"[StagedDiffusion] phase={self.phase}  dist={dist:.3f}  gripper={action[6] if action.shape[0]>=7 else 'n/a':.2f}", flush=True)
        return action


def load_policy_wrapper(
    checkpoint_path: str,
    device: torch.device,
    staged_diffusion: bool = False,
) -> Tuple[BasePolicyWrapper, PolicyInfo, dict]:
    """Load either diffusion or legacy MLP checkpoint and return inference wrapper.

    Args:
        staged_diffusion: When True, wraps diffusion checkpoints with
            StagedDiffusionPolicyWrapper for explicit approach/grasp/pull phases.
    """
    ckpt = _safe_torch_load(checkpoint_path, device)

    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    epoch = int(ckpt.get("epoch", 0))
    loss = float(ckpt.get("loss", float("nan")))

    checkpoint_type = ckpt.get("checkpoint_type", "legacy_mlp")

    state_keys = ckpt.get("state_keys")
    state_key_dims = ckpt.get("state_key_dims")

    if checkpoint_type == "staged_bc_handle_pull":
        model_kwargs = dict(ckpt.get("model_kwargs", {}))
        phase_cfg = dict(ckpt.get("phase_cfg", {}))
        model = StageConditionedBCNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(model_kwargs.get("hidden_dim", 256)),
            num_phases=int(model_kwargs.get("num_phases", 3)),
        ).to(device)
        model.load_state_dict(_unwrap_model_state_dict(ckpt["model_state_dict"]))
        model.eval()

        norm = ckpt.get("normalization", {})
        state_mean = np.asarray(norm.get("state_mean"), dtype=np.float32)
        state_std = np.asarray(norm.get("state_std"), dtype=np.float32)
        action_mean = np.asarray(norm.get("action_mean"), dtype=np.float32)
        action_std = np.asarray(norm.get("action_std"), dtype=np.float32)

        wrapper = StagedBCPolicyWrapper(
            model=model,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            state_keys=state_keys,
            state_key_dims=state_key_dims,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            approach_dist_threshold=float(phase_cfg.get("approach_dist_threshold", 0.08)),
            reapproach_dist_threshold=float(phase_cfg.get("reapproach_dist_threshold", 0.12)),
            grasp_steps=int(phase_cfg.get("grasp_steps", 8)),
            gripper_close_value=float(phase_cfg.get("gripper_close_value", -1.0)),
        )
        info = PolicyInfo(
            model_type="staged_bc_handle_pull",
            state_dim=state_dim,
            action_dim=action_dim,
            epoch=epoch,
            loss=loss,
        )
        return wrapper, info, ckpt

    if checkpoint_type == "diffusion_unet_lowdim":
        model_kwargs = dict(ckpt.get("model_kwargs", {}))
        diffusion_kwargs = dict(ckpt.get("diffusion_kwargs", {}))
        model_state_dict = _unwrap_model_state_dict(ckpt["model_state_dict"])

        model = ConditionalUnet1D(
            action_dim=action_dim,
            state_dim=state_dim,
            n_obs_steps=int(model_kwargs["n_obs_steps"]),
            horizon=int(model_kwargs["horizon"]),
            base_channels=int(model_kwargs.get("base_channels", 64)),
            channel_mults=tuple(model_kwargs.get("channel_mults", (1, 2))),
            num_res_blocks=int(model_kwargs.get("num_res_blocks", 1)),
            time_emb_dim=int(model_kwargs.get("time_emb_dim", 128)),
            cond_dim=int(model_kwargs.get("cond_dim", 256)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            num_phases=int(model_kwargs.get("num_phases", 1)),
        ).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        scheduler = DDPMScheduler(
            num_train_timesteps=int(diffusion_kwargs.get("num_train_timesteps", 100)),
            beta_start=float(diffusion_kwargs.get("beta_start", 1e-4)),
            beta_end=float(diffusion_kwargs.get("beta_end", 2e-2)),
        )
        policy = DiffusionPolicyCore(model=model, scheduler=scheduler).to(device)
        policy.eval()

        norm = ckpt.get("normalization", {})
        state_mean = np.asarray(norm.get("state_mean"), dtype=np.float32)
        state_std = np.asarray(norm.get("state_std"), dtype=np.float32)
        action_mean = np.asarray(norm.get("action_mean"), dtype=np.float32)
        action_std = np.asarray(norm.get("action_std"), dtype=np.float32)

        diffusion_wrapper = DiffusionPolicyWrapper(
            policy=policy,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            state_keys=state_keys,
            state_key_dims=state_key_dims,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            n_obs_steps=int(model_kwargs["n_obs_steps"]),
            n_action_steps=int(model_kwargs.get("n_action_steps", 2)),
            num_inference_steps=int(model_kwargs.get("num_inference_steps", 16)),
        )

        wrapper: BasePolicyWrapper
        if staged_diffusion:
            wrapper = StagedDiffusionPolicyWrapper(inner=diffusion_wrapper)
        else:
            wrapper = diffusion_wrapper

        info = PolicyInfo(
            model_type="diffusion_unet_lowdim",
            state_dim=state_dim,
            action_dim=action_dim,
            epoch=epoch,
            loss=loss,
        )
        return wrapper, info, ckpt

    if checkpoint_type == "temporal_unet_bc_lowdim":
        model_kwargs = dict(ckpt.get("model_kwargs", {}))
        model_state_dict = _unwrap_model_state_dict(ckpt["model_state_dict"])

        model = ConditionalUnet1D(
            action_dim=action_dim,
            state_dim=state_dim,
            n_obs_steps=int(model_kwargs["n_obs_steps"]),
            horizon=int(model_kwargs["horizon"]),
            base_channels=int(model_kwargs.get("base_channels", 64)),
            channel_mults=tuple(model_kwargs.get("channel_mults", (1, 2))),
            num_res_blocks=int(model_kwargs.get("num_res_blocks", 1)),
            time_emb_dim=int(model_kwargs.get("time_emb_dim", 128)),
            cond_dim=int(model_kwargs.get("cond_dim", 256)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            num_phases=int(model_kwargs.get("num_phases", 1)),
        ).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()

        norm = ckpt.get("normalization", {})
        state_mean = np.asarray(norm.get("state_mean"), dtype=np.float32)
        state_std = np.asarray(norm.get("state_std"), dtype=np.float32)
        action_mean = np.asarray(norm.get("action_mean"), dtype=np.float32)
        action_std = np.asarray(norm.get("action_std"), dtype=np.float32)

        wrapper = TemporalUnetBCPolicyWrapper(
            model=model,
            device=device,
            state_dim=state_dim,
            action_dim=action_dim,
            state_keys=state_keys,
            state_key_dims=state_key_dims,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
            n_obs_steps=int(model_kwargs["n_obs_steps"]),
            n_action_steps=int(model_kwargs.get("n_action_steps", 2)),
        )

        info = PolicyInfo(
            model_type="temporal_unet_bc_lowdim",
            state_dim=state_dim,
            action_dim=action_dim,
            epoch=epoch,
            loss=loss,
        )
        return wrapper, info, ckpt

    # Legacy MLP checkpoint support.
    model = SimplePolicyNet(state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(_unwrap_model_state_dict(ckpt["model_state_dict"]))
    model.eval()

    wrapper = SimplePolicyWrapper(
        model=model,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
        state_keys=state_keys,
        state_key_dims=state_key_dims,
    )
    info = PolicyInfo(
        model_type="simple_mlp_bc",
        state_dim=state_dim,
        action_dim=action_dim,
        epoch=epoch,
        loss=loss,
    )
    return wrapper, info, ckpt
