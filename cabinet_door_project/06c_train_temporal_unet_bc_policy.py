"""
Step 6c: Train Temporal BC Policy (Non-Staged, U-Net Backbone)
===============================================================
Trains a natural temporal behavior cloning policy using the same 1D Conv U-Net
architecture as diffusion, but with direct supervised action-sequence targets.

This trainer is explicitly non-staged (no phase runtime controller).
It auto-uses 05b augmented parquet data and can enforce that requirement.

Usage:
    python 06c_train_temporal_unet_bc_policy.py
    python 06c_train_temporal_unet_bc_policy.py --config configs/temporal_unet_bc_policy.yaml
    python 06c_train_temporal_unet_bc_policy.py --epochs 100
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import yaml

from policy_common import CabinetSequenceDataset, ConditionalUnet1D
from runtime_setup import select_torch_device


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, epoch, loss, model, optimizer, dataset, config):
    ckpt = {
        "checkpoint_type": "temporal_unet_bc_lowdim",
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_dim": int(dataset.state_dim),
        "action_dim": int(dataset.action_dim),
        "state_keys": list(dataset.state_keys),
        "state_key_dims": dict(dataset.state_key_dims),
        "normalization": {
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
        },
        "model_kwargs": {
            "horizon": int(config["horizon"]),
            "n_obs_steps": int(config["n_obs_steps"]),
            "n_action_steps": int(config["n_action_steps"]),
            "base_channels": int(config["base_channels"]),
            "channel_mults": list(config["channel_mults"]),
            "num_res_blocks": int(config["num_res_blocks"]),
            "time_emb_dim": int(config["time_emb_dim"]),
            "cond_dim": int(config["cond_dim"]),
            "dropout": float(config["dropout"]),
            "num_phases": 1,
        },
        "action_postprocess": {
            "dataset_to_env_remap": True,
            "gripper_threshold": 0.0,
            "base_mode_threshold": 0.0,
        },
        "config": dict(config),
    }

    import torch

    torch.save(ckpt, path)


def _checkpoint_names(config):
    best_name = str(config.get("best_checkpoint_name", "best_policy.pt"))
    final_name = str(config.get("final_checkpoint_name", "final_policy.pt"))
    return best_name, final_name


def _next_cycled_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def _split_episodes(
    num_episodes: int,
    val_ratio: float,
    seed: int,
    min_val_episodes: int = 1,
) -> Tuple[List[int], List[int]]:
    if num_episodes < 2:
        return list(range(num_episodes)), []

    n_val = int(round(float(val_ratio) * num_episodes))
    n_val = max(int(min_val_episodes), n_val)
    n_val = min(max(1, num_episodes - 1), n_val)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(num_episodes).tolist()
    val_eps = sorted(perm[:n_val])
    train_eps = sorted(perm[n_val:])
    return train_eps, val_eps


def _subset_indices_by_episode(
    dataset: CabinetSequenceDataset,
    episode_ids: Sequence[int],
    burn_in_steps: int = 0,
) -> List[int]:
    ep_set = set(int(e) for e in episode_ids)
    min_t = int(dataset.n_obs_steps - 1 + max(0, int(burn_in_steps)))
    indices: List[int] = []
    for i, (ep_idx, t) in enumerate(dataset.sample_index):
        if ep_idx in ep_set and t >= min_t:
            indices.append(i)
    return indices


def train_temporal_unet_bc(config):
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Subset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    set_seed(int(config["seed"]))

    print_section("Temporal BC Policy (Non-Staged, U-Net)")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    print("\nLoading sequence dataset...")
    dataset = CabinetSequenceDataset(
        dataset_path=dataset_path,
        horizon=int(config["horizon"]),
        n_obs_steps=int(config["n_obs_steps"]),
        include_augmented=bool(config["include_augmented"]),
        include_all_lowdim=bool(config["include_all_lowdim"]),
        use_observation_state=bool(config.get("use_observation_state", True)),
        state_keys_override=config.get("state_keys_override"),
        max_episodes=(
            None if config.get("max_episodes") in [None, 0] else int(config["max_episodes"])
        ),
    )

    if bool(config.get("require_augmented", True)) and dataset.source != "augmented":
        raise RuntimeError(
            "Augmented data was required, but trainer loaded raw parquet files.\n"
            "Run: python 05b_augment_handle_data.py\n"
            "If you intentionally want raw data, use --allow_raw."
        )

    print(f"  Data source:        {dataset.source}")
    if dataset.source == "augmented":
        print("  Augmented features: enabled (verified)")
    print(f"  Episodes loaded:    {len(dataset.episodes_states)}")
    print(f"  Training windows:   {len(dataset)}")
    print(f"  State dim:          {dataset.state_dim}")
    print(f"  Action dim:         {dataset.action_dim}")
    print(f"  State keys:         {dataset.state_keys}")

    expected_state_dim = config.get("expected_state_dim")
    if expected_state_dim is not None and int(expected_state_dim) != int(dataset.state_dim):
        raise RuntimeError(
            f"State dim mismatch: expected {int(expected_state_dim)}, got {dataset.state_dim}. "
            "Check state_keys_override / augmentation columns."
        )

    train_eps, val_eps = _split_episodes(
        num_episodes=len(dataset.episodes_states),
        val_ratio=float(config.get("val_split_ratio", 0.15)),
        seed=int(config["seed"]),
        min_val_episodes=int(config.get("min_val_episodes", 1)),
    )

    train_ids = _subset_indices_by_episode(
        dataset,
        train_eps,
        burn_in_steps=int(config.get("episode_burn_in_steps", 0)),
    )
    val_ids = _subset_indices_by_episode(
        dataset,
        val_eps,
        burn_in_steps=int(config.get("episode_burn_in_steps", 0)),
    )

    if not train_ids:
        raise RuntimeError("No train windows available after split.")

    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids) if val_ids else None

    print(f"  Train episodes:     {len(train_eps)}")
    print(f"  Val episodes:       {len(val_eps)}")
    print(f"  Train windows:      {len(train_subset)}")
    print(f"  Val windows:        {len(val_subset) if val_subset is not None else 0}")

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=(len(train_subset) >= batch_size),
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    if len(train_loader) == 0:
        raise RuntimeError("No batches available. Reduce batch_size or load more episodes.")

    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader = DataLoader(
            val_subset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=int(config["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )

    steps_per_epoch = int(config.get("steps_per_epoch", 0))
    if steps_per_epoch <= 0:
        steps_per_epoch = len(train_loader)

    device = select_torch_device()
    print(f"\nDevice: {device}")

    model = ConditionalUnet1D(
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        n_obs_steps=int(config["n_obs_steps"]),
        horizon=int(config["horizon"]),
        base_channels=int(config["base_channels"]),
        channel_mults=tuple(config["channel_mults"]),
        num_res_blocks=int(config["num_res_blocks"]),
        time_emb_dim=int(config["time_emb_dim"]),
        cond_dim=int(config["cond_dim"]),
        dropout=float(config["dropout"]),
        num_phases=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    use_amp = bool(config.get("use_amp", True)) and getattr(device, "type", "") == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    state_mean = torch.from_numpy(dataset.state_mean).to(device).view(1, 1, -1)
    state_std = torch.from_numpy(dataset.state_std).to(device).view(1, 1, -1)
    action_mean = torch.from_numpy(dataset.action_mean).to(device).view(1, 1, -1)
    action_std = torch.from_numpy(dataset.action_std).to(device).view(1, 1, -1)

    print(f"Model params:       {count_parameters(model):,}")

    print_section("Training")
    print(f"Epochs:             {config['epochs']}")
    print(f"Batch size:         {config['batch_size']}")
    print(f"LR:                 {config['learning_rate']}")
    print(f"Action horizon:     {config['horizon']}")
    print(f"Obs steps:          {config['n_obs_steps']}")
    print(f"Steps / epoch:      {steps_per_epoch}")
    print(f"AMP:                {'on' if use_amp else 'off'}")
    print(f"Val split ratio:    {config.get('val_split_ratio', 0.15)}")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_name, final_name = _checkpoint_names(config)

    best_metric = float("inf")
    best_epoch = 0
    avg_loss = float("inf")
    avg_val_loss = float("inf")
    epochs_without_improve = 0
    early_stop_patience = int(config.get("early_stopping_patience", 25))
    early_stop_min_delta = float(config.get("early_stopping_min_delta", 1e-4))
    loader_iter = iter(train_loader)

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            (obs_seq, action_seq, _phase), loader_iter = _next_cycled_batch(
                train_loader, loader_iter
            )

            obs_seq = obs_seq.to(device)
            action_seq = action_seq.to(device)

            obs_norm = (obs_seq - state_mean) / state_std
            action_norm = (action_seq - action_mean) / action_std

            zero_actions = torch.zeros_like(action_norm)
            zero_t = torch.zeros((action_norm.shape[0],), device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_action_norm = model(
                    noisy_action=zero_actions,
                    timesteps=zero_t,
                    obs_seq=obs_norm,
                    phase=None,
                )
                loss_mse = F.mse_loss(pred_action_norm, action_norm)
                loss_l1 = F.l1_loss(pred_action_norm, action_norm)
                if pred_action_norm.shape[1] > 1:
                    pred_vel = pred_action_norm[:, 1:] - pred_action_norm[:, :-1]
                    tgt_vel = action_norm[:, 1:] - action_norm[:, :-1]
                    loss_vel = F.mse_loss(pred_vel, tgt_vel)
                else:
                    loss_vel = torch.zeros((), device=device)

                loss = (
                    loss_mse
                    + float(config["l1_weight"]) * loss_l1
                    + float(config["velocity_weight"]) * loss_vel
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if float(config["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for obs_seq, action_seq, _phase in val_loader:
                    obs_seq = obs_seq.to(device)
                    action_seq = action_seq.to(device)

                    obs_norm = (obs_seq - state_mean) / state_std
                    action_norm = (action_seq - action_mean) / action_std
                    zero_actions = torch.zeros_like(action_norm)
                    zero_t = torch.zeros((action_norm.shape[0],), device=device, dtype=torch.long)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred_action_norm = model(
                            noisy_action=zero_actions,
                            timesteps=zero_t,
                            obs_seq=obs_norm,
                            phase=None,
                        )
                        loss_mse = F.mse_loss(pred_action_norm, action_norm)
                        loss_l1 = F.l1_loss(pred_action_norm, action_norm)
                        if pred_action_norm.shape[1] > 1:
                            pred_vel = pred_action_norm[:, 1:] - pred_action_norm[:, :-1]
                            tgt_vel = action_norm[:, 1:] - action_norm[:, :-1]
                            loss_vel = F.mse_loss(pred_vel, tgt_vel)
                        else:
                            loss_vel = torch.zeros((), device=device)

                        val_loss = (
                            loss_mse
                            + float(config["l1_weight"]) * loss_l1
                            + float(config["velocity_weight"]) * loss_vel
                        )

                    val_loss_total += float(val_loss.item())
                    val_batches += 1

            avg_val_loss = val_loss_total / max(val_batches, 1)
            monitor_metric = avg_val_loss
            monitor_name = "Val"
        else:
            avg_val_loss = float("nan")
            monitor_metric = avg_loss
            monitor_name = "Train"

        dt = time.time() - t0

        if (epoch + 1) % int(config["log_every"]) == 0 or epoch == 0:
            if val_loader is not None:
                print(
                    f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                    f"Train: {avg_loss:.6f}  Val: {avg_val_loss:.6f}  ({dt:.2f}s)"
                )
            else:
                print(
                    f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                    f"Loss: {avg_loss:.6f}  ({dt:.2f}s)"
                )

        if monitor_metric < (best_metric - early_stop_min_delta):
            best_metric = monitor_metric
            best_epoch = epoch + 1
            epochs_without_improve = 0
            save_checkpoint(
                path=str(checkpoint_dir / best_name),
                epoch=epoch + 1,
                loss=best_metric,
                model=model,
                optimizer=optimizer,
                dataset=dataset,
                config=config,
            )
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(best {monitor_name.lower()}={best_metric:.6f} at epoch {best_epoch})"
            )
            break

    save_checkpoint(
        path=str(checkpoint_dir / final_name),
        epoch=epoch + 1,
        loss=avg_loss,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
    )

    print("\nTraining complete!")
    if val_loader is not None:
        print(f"Best val loss:      {best_metric:.6f} (epoch {best_epoch})")
    else:
        print(f"Best loss:          {best_metric:.6f}")
    print(f"Best checkpoint:    {checkpoint_dir / best_name}")
    print(f"Final checkpoint:   {checkpoint_dir / final_name}")


def default_config():
    return {
        "seed": 42,
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "num_workers": 0,
        "grad_clip": 1.0,
        "log_every": 1,
        "checkpoint_dir": "/tmp/cabinet_temporal_unet_bc_working_ckpts",
        "best_checkpoint_name": "best_policy.pt",
        "final_checkpoint_name": "final_policy.pt",
        "max_episodes": None,
        "include_augmented": True,
        "require_augmented": True,
        "include_all_lowdim": False,
        "use_observation_state": True,
        "state_keys_override": [
            "observation.state",
            "observation.handle_pos",
            "observation.handle_to_eef_pos",
        ],
        "expected_state_dim": 22,
        "horizon": 16,
        "n_obs_steps": 16,
        "n_action_steps": 8,
        "steps_per_epoch": 0,
        "episode_burn_in_steps": 0,
        "val_split_ratio": 0.15,
        "min_val_episodes": 1,
        "early_stopping_patience": 25,
        "early_stopping_min_delta": 1e-4,
        "use_amp": True,
        "base_channels": 64,
        "channel_mults": [1, 2],
        "num_res_blocks": 1,
        "time_emb_dim": 128,
        "cond_dim": 256,
        "dropout": 0.0,
        "l1_weight": 0.10,
        "velocity_weight": 0.10,
    }


def merge_config(base, override):
    merged = dict(base)
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def merge_config_allow_none(base, override):
    """Merge config keys exactly, allowing explicit nulls from YAML."""
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Train a non-staged temporal U-Net behavior cloning policy"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_split_ratio", type=float, default=None)
    parser.add_argument(
        "--allow_raw",
        action="store_true",
        help="Allow training on raw parquet data (disables augmented-data requirement)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Temporal U-Net BC Training")
    print("=" * 60)

    cfg = default_config()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"ERROR: Config file not found: {cfg_path}")
            sys.exit(1)
        cfg = merge_config_allow_none(cfg, load_config(str(cfg_path)))

    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "checkpoint_dir": args.checkpoint_dir,
        "max_episodes": args.max_episodes,
        "steps_per_epoch": args.steps_per_epoch,
        "num_workers": args.num_workers,
        "early_stopping_patience": args.patience,
        "val_split_ratio": args.val_split_ratio,
    }
    cfg = merge_config(cfg, cli_overrides)
    if args.allow_raw:
        cfg["require_augmented"] = False

    train_temporal_unet_bc(cfg)


if __name__ == "__main__":
    main()
