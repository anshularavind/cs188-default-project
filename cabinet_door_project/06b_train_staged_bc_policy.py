"""
Step 6b: Train Staged BC Policy (Approach -> Grasp -> Pull)
============================================================
Trains a phase-conditioned behavior cloning policy that explicitly follows:
1) approach the handle,
2) close the gripper,
3) pull the door.

Usage:
    python 06b_train_staged_bc_policy.py
    python 06b_train_staged_bc_policy.py --config configs/staged_bc_policy.yaml
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from policy_common import CabinetSequenceDataset, StageConditionedBCNet
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


def find_state_slice(state_keys, state_key_dims, target_keys):
    targets = set(target_keys)
    offset = 0
    for key in state_keys:
        dim = int(state_key_dims.get(key, 0))
        stripped = key[len("observation.") :] if key.startswith("observation.") else key
        if key in targets or stripped in targets:
            return slice(offset, offset + dim)
        offset += dim
    return None


def label_episode_phases(
    states,
    state_keys,
    state_key_dims,
    approach_dist_threshold,
    grasp_steps,
    pull_open_threshold,
):
    handle_slice = find_state_slice(
        state_keys,
        state_key_dims,
        ["observation.handle_to_eef_pos", "handle_to_eef_pos"],
    )
    door_slice = find_state_slice(
        state_keys,
        state_key_dims,
        ["observation.door_openness", "door_openness"],
    )

    T = states.shape[0]
    phases = np.zeros((T,), dtype=np.int64)

    if handle_slice is not None and (handle_slice.stop - handle_slice.start) >= 3:
        h = states[:, handle_slice.start : handle_slice.start + 3]
        dist = np.linalg.norm(h, axis=1)
    else:
        dist = np.full((T,), np.inf, dtype=np.float32)

    if door_slice is not None and (door_slice.stop - door_slice.start) >= 1:
        door_open = states[:, door_slice.start]
    else:
        door_open = np.zeros((T,), dtype=np.float32)

    near_idxs = np.where(dist <= approach_dist_threshold)[0]
    first_near = int(near_idxs[0]) if len(near_idxs) > 0 else int(T // 3)

    for t in range(T):
        if t < first_near:
            phase = 0
        elif t < first_near + grasp_steps and door_open[t] < pull_open_threshold:
            phase = 1
        else:
            phase = 2

        # If we drift far from handle before opening, re-label as approach.
        if dist[t] > approach_dist_threshold * 1.6 and door_open[t] < pull_open_threshold * 0.5:
            phase = 0

        phases[t] = phase

    return phases


def build_transition_data(dataset, config):
    states_all = []
    actions_all = []
    phases_all = []

    for states, actions in zip(dataset.episodes_states, dataset.episodes_actions):
        T = min(len(states), len(actions))
        if T <= 0:
            continue
        states = states[:T]
        actions = actions[:T]

        phases = label_episode_phases(
            states=states,
            state_keys=dataset.state_keys,
            state_key_dims=dataset.state_key_dims,
            approach_dist_threshold=float(config["approach_dist_threshold"]),
            grasp_steps=int(config["grasp_steps"]),
            pull_open_threshold=float(config["pull_open_threshold"]),
        )

        states_all.append(states)
        actions_all.append(actions)
        phases_all.append(phases)

    if not states_all:
        raise RuntimeError("No transitions found for staged BC training")

    states_arr = np.concatenate(states_all, axis=0).astype(np.float32)
    actions_arr = np.concatenate(actions_all, axis=0).astype(np.float32)
    phases_arr = np.concatenate(phases_all, axis=0).astype(np.int64)

    return states_arr, actions_arr, phases_arr


def save_checkpoint(
    path,
    epoch,
    loss,
    model,
    optimizer,
    dataset,
    config,
    state_mean,
    state_std,
    action_mean,
    action_std,
    gripper_close_value,
):
    ckpt = {
        "checkpoint_type": "staged_bc_handle_pull",
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_dim": int(dataset.state_dim),
        "action_dim": int(dataset.action_dim),
        "state_keys": list(dataset.state_keys),
        "state_key_dims": dict(dataset.state_key_dims),
        "normalization": {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
        },
        "model_kwargs": {
            "hidden_dim": int(config["hidden_dim"]),
            "num_phases": 3,
        },
        "phase_cfg": {
            "approach_dist_threshold": float(config["approach_dist_threshold"]),
            "reapproach_dist_threshold": float(config["reapproach_dist_threshold"]),
            "grasp_steps": int(config["grasp_steps"]),
            "pull_open_threshold": float(config["pull_open_threshold"]),
            "gripper_close_value": float(gripper_close_value),
        },
        "config": dict(config),
    }

    import torch

    torch.save(ckpt, path)


def train_staged_bc(config):
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    set_seed(int(config["seed"]))

    print_section("Staged BC Policy (Approach -> Grasp -> Pull)")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    print("\nLoading dataset...")
    dataset = CabinetSequenceDataset(
        dataset_path=dataset_path,
        horizon=2,
        n_obs_steps=1,
        include_augmented=bool(config["include_augmented"]),
        include_all_lowdim=bool(config["include_all_lowdim"]),
        use_observation_state=bool(config.get("use_observation_state", True)),
        max_episodes=(
            None if config.get("max_episodes") in [None, 0] else int(config["max_episodes"])
        ),
    )

    if bool(config.get("require_augmented", True)) and dataset.source != "augmented":
        raise RuntimeError(
            "Augmented data was required, but trainer loaded raw parquet files.\n"
            "Run: python 05b_augment_handle_data.py\n"
            "If you intentionally want raw data, set require_augmented=false."
        )

    print(f"  Data source:        {dataset.source}")
    print(f"  Episodes loaded:    {len(dataset.episodes_states)}")
    print(f"  State dim:          {dataset.state_dim}")
    print(f"  Action dim:         {dataset.action_dim}")
    print(f"  State keys:         {dataset.state_keys}")

    states, actions, phases = build_transition_data(dataset, config)
    print(f"  Transitions:        {len(states)}")

    counts = np.bincount(phases, minlength=3)
    print(
        f"  Phase counts:       approach={counts[0]} grasp={counts[1]} pull={counts[2]}"
    )

    if dataset.action_dim >= 7 and counts[1] > 0:
        gripper_close_value = float(np.median(actions[phases == 1, 6]))
    else:
        gripper_close_value = -1.0
    print(f"  Gripper close cmd:  {gripper_close_value:+.3f}")

    state_mean = states.mean(axis=0).astype(np.float32)
    state_std = np.clip(states.std(axis=0).astype(np.float32), 1e-3, None)
    action_mean = actions.mean(axis=0).astype(np.float32)
    action_std = np.clip(actions.std(axis=0).astype(np.float32), 1e-3, None)

    states_t = torch.from_numpy(states)
    actions_t = torch.from_numpy(actions)
    phases_t = torch.from_numpy(phases)

    ds = TensorDataset(states_t, actions_t, phases_t)
    loader = DataLoader(
        ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    device = select_torch_device()
    print(f"\nDevice: {device}")

    model = StageConditionedBCNet(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_phases=3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    phase_weights = torch.tensor(config["phase_weights"], dtype=torch.float32, device=device)
    s_mean_t = torch.from_numpy(state_mean).to(device)
    s_std_t = torch.from_numpy(state_std).to(device)
    a_mean_t = torch.from_numpy(action_mean).to(device)
    a_std_t = torch.from_numpy(action_std).to(device)

    handle_slice = find_state_slice(
        dataset.state_keys,
        dataset.state_key_dims,
        ["observation.handle_to_eef_pos", "handle_to_eef_pos"],
    )

    print(f"Model params: {count_parameters(model):,}")

    print_section("Training")
    print(f"Epochs:            {config['epochs']}")
    print(f"Batch size:        {config['batch_size']}")
    print(f"Learning rate:     {config['learning_rate']}")

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for s_b, a_b, p_b in loader:
            s_b = s_b.to(device)
            a_b = a_b.to(device)
            p_b = p_b.to(device)

            s_norm = (s_b - s_mean_t) / s_std_t
            a_norm = (a_b - a_mean_t) / a_std_t

            pred_norm = model(s_norm, p_b)
            per_item = ((pred_norm - a_norm) ** 2).mean(dim=1)
            loss = (per_item * phase_weights[p_b]).mean()

            pred_act = pred_norm * a_std_t + a_mean_t

            if handle_slice is not None and (handle_slice.stop - handle_slice.start) >= 3:
                idx = p_b == 0
                if torch.any(idx):
                    h = s_b[idx, handle_slice.start : handle_slice.start + 3]
                    target_xyz = torch.clamp(0.8 * h, -0.2, 0.2)
                    loss = loss + float(config["approach_aux_weight"]) * F.mse_loss(
                        pred_act[idx, :3],
                        target_xyz,
                    )

            if dataset.action_dim >= 7:
                idx = p_b == 1
                if torch.any(idx):
                    target_grip = torch.full_like(pred_act[idx, 6], gripper_close_value)
                    loss = loss + float(config["grasp_aux_weight"]) * F.mse_loss(
                        pred_act[idx, 6],
                        target_grip,
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
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
                model=model,
                optimizer=optimizer,
                dataset=dataset,
                config=config,
                state_mean=state_mean,
                state_std=state_std,
                action_mean=action_mean,
                action_std=action_std,
                gripper_close_value=gripper_close_value,
            )

    save_checkpoint(
        path=str(checkpoint_dir / "final_policy.pt"),
        epoch=int(config["epochs"]),
        loss=avg_loss,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        gripper_close_value=gripper_close_value,
    )

    print("\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {checkpoint_dir / 'best_policy.pt'}")
    print(f"Final checkpoint: {checkpoint_dir / 'final_policy.pt'}")


def default_config():
    return {
        "seed": 42,
        "epochs": 40,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "weight_decay": 1e-6,
        "num_workers": 0,
        "grad_clip": 1.0,
        "log_every": 5,
        "checkpoint_dir": "/tmp/cabinet_stage_bc_checkpoints",
        "max_episodes": None,
        "include_augmented": True,
        "require_augmented": True,
        "include_all_lowdim": True,
        "use_observation_state": False,
        "hidden_dim": 256,
        "approach_dist_threshold": 0.08,
        "reapproach_dist_threshold": 0.13,
        "grasp_steps": 8,
        "pull_open_threshold": 0.08,
        "phase_weights": [1.6, 1.3, 1.0],
        "approach_aux_weight": 0.2,
        "grasp_aux_weight": 0.2,
    }


def merge_config(base, override):
    merged = dict(base)
    for k, v in override.items():
        if v is not None:
            merged[k] = v
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train staged BC policy for OpenCabinet")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Staged BC Training")
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

    train_staged_bc(cfg)


if __name__ == "__main__":
    main()
