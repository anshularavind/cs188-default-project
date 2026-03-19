"""
Step 6: Train a Low-Dim Diffusion Policy (1D Conv U-Net)
===========================================================
Trains a state-based diffusion policy for OpenCabinet using a 1D Conv U-Net
backbone, following the instructor recommendation.

This script auto-detects and uses 05b augmented parquet files when available.

Usage:
    python 06_train_policy.py
    python 06_train_policy.py --config configs/diffusion_policy.yaml
    python 06_train_policy.py --epochs 150 --batch_size 128
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from policy_common import (
    CabinetSequenceDataset,
    ConditionalUnet1D,
    DDPMScheduler,
    DiffusionPolicyCore,
)
from runtime_setup import select_torch_device


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
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


def save_checkpoint(
    path,
    epoch,
    loss,
    model,
    optimizer,
    dataset,
    config,
):
    ckpt = {
        "checkpoint_type": "diffusion_unet_lowdim",
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
            "num_inference_steps": int(config["num_inference_iters"]),
            "base_channels": int(config["base_channels"]),
            "channel_mults": list(config["channel_mults"]),
            "num_res_blocks": int(config["num_res_blocks"]),
            "time_emb_dim": int(config["time_emb_dim"]),
            "cond_dim": int(config["cond_dim"]),
            "dropout": float(config["dropout"]),
        },
        "diffusion_kwargs": {
            "num_train_timesteps": int(config["num_diffusion_iters"]),
            "beta_start": float(config["beta_start"]),
            "beta_end": float(config["beta_end"]),
        },
        "config": dict(config),
    }
    import torch

    torch.save(ckpt, path)


def train_diffusion_policy(config):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    set_seed(int(config["seed"]))

    print_section("Low-Dim Diffusion Policy (1D Conv U-Net)")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    print("\nLoading sequence dataset...")
    dataset = CabinetSequenceDataset(
        dataset_path=dataset_path,
        horizon=int(config["horizon"]),
        n_obs_steps=int(config["n_obs_steps"]),
        include_augmented=bool(config["include_augmented"]),
        include_all_lowdim=bool(config["include_all_lowdim"]),
        max_episodes=(
            None if config.get("max_episodes") in [None, 0] else int(config["max_episodes"])
        ),
    )

    if bool(config.get("require_augmented", True)) and dataset.source != "augmented":
        raise RuntimeError(
            "Augmented data was required, but trainer loaded raw parquet files.\\n"
            "Run: python 05b_augment_handle_data.py\\n"
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

    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    device = select_torch_device()
    print(f"\nDevice: {device}")

    unet = ConditionalUnet1D(
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
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=int(config["num_diffusion_iters"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    )

    policy = DiffusionPolicyCore(model=unet, scheduler=scheduler).to(device)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    # Normalization tensors for fast broadcasting in training loop.
    state_mean = torch.from_numpy(dataset.state_mean).to(device).view(1, 1, -1)
    state_std = torch.from_numpy(dataset.state_std).to(device).view(1, 1, -1)
    action_mean = torch.from_numpy(dataset.action_mean).to(device).view(1, 1, -1)
    action_std = torch.from_numpy(dataset.action_std).to(device).view(1, 1, -1)

    print(f"Model params: {count_parameters(policy):,}")

    print_section("Training")
    print(f"Epochs:            {config['epochs']}")
    print(f"Batch size:        {config['batch_size']}")
    print(f"LR:                {config['learning_rate']}")
    print(f"Diffusion steps:   {config['num_diffusion_iters']}")
    print(f"Action horizon:    {config['horizon']}")
    print(f"Obs steps:         {config['n_obs_steps']}")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")

    for epoch in range(int(config["epochs"])):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        for obs_seq, action_seq in dataloader:
            obs_seq = obs_seq.to(device)
            action_seq = action_seq.to(device)

            obs_norm = (obs_seq - state_mean) / state_std
            action_norm = (action_seq - action_mean) / action_std

            timesteps = torch.randint(
                low=0,
                high=int(config["num_diffusion_iters"]),
                size=(action_norm.shape[0],),
                device=device,
            )

            noise = torch.randn_like(action_norm)
            noisy_action = scheduler.add_noise(action_norm, noise, timesteps)

            pred_noise = policy(noisy_action=noisy_action, timesteps=timesteps, obs_seq=obs_norm)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["grad_clip"]))
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % int(config["log_every"]) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                path=str(checkpoint_dir / "best_policy.pt"),
                epoch=epoch + 1,
                loss=best_loss,
                model=policy,
                optimizer=optimizer,
                dataset=dataset,
                config=config,
            )

    save_checkpoint(
        path=str(checkpoint_dir / "final_policy.pt"),
        epoch=int(config["epochs"]),
        loss=avg_loss,
        model=policy,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
    )

    print("\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {checkpoint_dir / 'best_policy.pt'}")
    print(f"Final checkpoint: {checkpoint_dir / 'final_policy.pt'}")


def default_config():
    return {
        "seed": 42,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "num_workers": 0,
        "grad_clip": 1.0,
        "log_every": 5,
        "checkpoint_dir": "/tmp/cabinet_policy_checkpoints",
        "max_episodes": None,
        "include_augmented": True,
        "require_augmented": True,
        "include_all_lowdim": False,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "num_diffusion_iters": 100,
        "num_inference_iters": 16,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "base_channels": 128,
        "channel_mults": [1, 2, 4],
        "num_res_blocks": 2,
        "time_emb_dim": 128,
        "cond_dim": 512,
        "dropout": 0.0,
    }


def merge_config(base, override):
    merged = dict(base)
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion policy for OpenCabinet")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument(
        "--allow_raw",
        action="store_true",
        help="Allow training on raw parquet data (disables augmented-data requirement)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Diffusion Policy Training")
    print("=" * 60)

    cfg = default_config()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"ERROR: Config file not found: {cfg_path}")
            sys.exit(1)
        cfg = merge_config(cfg, load_config(str(cfg_path)))

    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "checkpoint_dir": args.checkpoint_dir,
        "max_episodes": args.max_episodes,
    }
    cfg = merge_config(cfg, cli_overrides)
    if args.allow_raw:
        cfg["require_augmented"] = False

    train_diffusion_policy(cfg)


if __name__ == "__main__":
    main()
