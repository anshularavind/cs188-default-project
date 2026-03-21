# Cabinet Door Project - Current Working Guide

This README reflects the **current codebase** and the policy setup we finalized:

- Primary training script: `06c_train_temporal_unet_bc_policy.py`
- Primary config (best run setup): `configs/temporal_unet_bc_old_exact_400_policy3.yaml`
- Primary checkpoint names: `best_policy_3.pt`, `final_policy_3.pt`
- Success rule in eval/visualization: **any one cabinet door open** (default threshold `0.10`)

## Recommended Workflow (Local)

From repo root:

```bash
./install.sh
source .venv/bin/activate
cd cabinet_door_project
python 00_verify_installation.py
```

If verify passes, run the actual training pipeline:

```bash
python 04_download_dataset.py
python 05b_augment_handle_data.py
python 06c_train_temporal_unet_bc_policy.py --config configs/temporal_unet_bc_old_exact_400_policy3.yaml
```

This writes checkpoints to:

- `/tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt`
- `/tmp/cabinet_temporal_unet_bc_ckpts/final_policy_3.pt`

In our recorded `best_policy_3` run, the best checkpoint was saved at **epoch 335** with
training loss **0.1012389**.

## How To Train The Correct Model (`06c` + Best Config)

Use exactly:

```bash
python 06c_train_temporal_unet_bc_policy.py --config configs/temporal_unet_bc_old_exact_400_policy3.yaml
```

### What this config does

- Temporal BC with 1D U-Net (non-staged)
- Uses **augmented dataset** (`include_augmented=true`, `require_augmented=true`)
- 400 epochs, batch size 512
- 11D low-dimensional state, action horizon 16
- Composite loss:
  \[
  \mathcal{L} = \mathcal{L}_{MSE} + 0.25\,\mathcal{L}_{L1} + 0.15\,\mathcal{L}_{vel}
  \]

Where velocity loss compares temporal differences:
\(\Delta a_t = a_t - a_{t-1}\).

## Evaluate Policy

```bash
python 07_evaluate_policy.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --num_rollouts 20 \
  --split pretrain
```

Notes:

- `--split pretrain` = in-distribution scenes
- `--split target` = held-out / harder generalization scenes
- `--seed` now defaults to `None` (randomized layout/style sequence each run)
- pass `--seed 123` if you want deterministic repeatability

You can loosen/tighten success threshold:

```bash
--success_threshold 0.23
```

## Visualize Policy

### On-screen

```bash
python 08_visualize_policy_rollout.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --num_episodes 1 \
  --max_steps 500
```

### Off-screen video

```bash
python 08_visualize_policy_rollout.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --offscreen \
  --video_path /tmp/policy_rollout.mp4 \
  --num_episodes 1 \
  --max_steps 500
```

### Force a specific layout/style

```bash
python 08_visualize_policy_rollout.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --layout_id 45 \
  --style_id 47
```

Important:

- `--layout_id` and `--style_id` must be passed together.
- Some layout/style combinations are invalid and can fail environment placement.
- If that happens, use a known valid pair from your eval logs.

## Method Summary (Project)

1. Download OpenCabinet demonstrations (`04_download_dataset.py`).
2. Add handle-centric augmentation columns (`05b_augment_handle_data.py`).
3. Train non-staged temporal U-Net BC (`06c_train_temporal_unet_bc_policy.py`).
4. Save best and final checkpoints based on monitored loss.
5. Evaluate and visualize with one-door-open success criterion.

## File-by-File Guide

### Repo root

- `README.md`: This document.
- `install.sh`: Local install/bootstrap script (venv, deps, assets flow).
- `pyproject.toml`: Python project metadata.
- `.python-version`: Python version hint.
- `main.py`: Minimal entry script.
- `.gitignore`: Git ignore rules (important for checkpoints/assets).
- `modelarch.txt`: Scratch file for architecture notes/diagrams.
- `robosuite/`: robosuite source checkout.
- `robocasa/`: robocasa source checkout.

### `cabinet_door_project/`

- `00_verify_installation.py`: End-to-end environment/dependency sanity check.
- `01_explore_environment.py`: Prints observation/action/task structure for OpenCabinet.
- `02_random_rollouts.py`: Random-policy rollouts for quick environment sanity.
- `03_teleop_collect_demos.py`: Teleoperation script to collect human demos.
- `04_download_dataset.py`: Downloads the OpenCabinet dataset in LeRobot format.
- `05_playback_demonstrations.py`: Playback utility for demonstration inspection.
- `05b_augment_handle_data.py`: Adds augmented handle features to parquet data.
- `06_train_policy.py`: Diffusion policy trainer (U-Net diffusion baseline).
- `06b_train_staged_bc_policy.py`: Staged BC trainer (approach/grasp/pull phases).
- `06c_train_temporal_unet_bc_policy.py`: **Primary trainer** (non-staged temporal U-Net BC).
- `07_evaluate_policy.py`: Batch evaluation script with per-episode layout/style logging.
- `08_visualize_policy_rollout.py`: On-screen/off-screen rollout visualizer with debug metrics and optional forced scene ids.
- `99_colab_setup.py`: Colab bootstrap utility (apt/pip/editable installs, optional assets/dataset/verify).
- `policy_common.py`: Shared core module:
  - dataset loader and state construction
  - 1D conditional U-Net
  - diffusion scheduler/core
  - BC/diffusion wrappers
  - action remap + binarization postprocess
  - checkpoint loading across policy types
- `runtime_setup.py`: Rendering backend and torch device selection helpers.
- `notebook.ipynb`: Multi-cell notebook workflow.
- `colab_one_cell.ipynb`: Single-cell Colab workflow.
- `configs/`: YAML config presets.
- `diffusion_ckpts/`: Local diffusion checkpoint storage (do not commit large files).
- `MUJOCO_LOG.TXT`: MuJoCo runtime log output.
- `__pycache__/`: Python bytecode cache.

### `cabinet_door_project/configs/`

- `diffusion_policy.yaml`: Diffusion U-Net baseline config (`06_train_policy.py`).
- `staged_bc_policy.yaml`: Staged BC config (`06b_train_staged_bc_policy.py`).
- `temporal_unet_bc_policy.yaml`: 22D working temporal U-Net BC config with val split.
- `temporal_unet_bc_old_exact.yaml`: Old-exact 11D temporal BC config (`best-policy_2.pt` / `final_policy_2.pt`).
- `temporal_unet_bc_old_exact_400_policy3.yaml`: **Best-policy-3 training config** (`best_policy_3.pt` / `final_policy_3.pt`).

## Policy Types Supported By Loader

`policy_common.load_policy_wrapper(...)` can load:

- `temporal_unet_bc_lowdim` (06c)
- `diffusion_unet_lowdim` (06)
- `staged_bc_handle_pull` (06b)
- legacy MLP checkpoints

## Colab Notes

Use `99_colab_setup.py` for base install, then run project scripts.
Kitchen asset download can be large and sometimes flaky; if it fails, rerun asset commands directly.
Checkpoints under `/tmp` are ephemeral in Colab runtime and should be copied to Drive if needed.

### Minimal Colab sequence

```python
REPO_DIR = "/content/cs188-default-project"
%cd /content
!git clone <your-repo-url> {REPO_DIR}
%cd {REPO_DIR}

# Base bootstrap (apt + pip + editable installs)
!python cabinet_door_project/99_colab_setup.py

# Optional but commonly needed for full RoboCasa scenes
!python robocasa/robocasa/scripts/download_kitchen_assets.py
!python robocasa/robocasa/scripts/setup_macros.py

%cd {REPO_DIR}/cabinet_door_project
!python 04_download_dataset.py
!python 05b_augment_handle_data.py
!python 06c_train_temporal_unet_bc_policy.py --config configs/temporal_unet_bc_old_exact_400_policy3.yaml
!python 07_evaluate_policy.py --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt --num_rollouts 20 --split pretrain
```

## Troubleshooting

- `No private macro file found`: run macro setup script from robocasa/robosuite as suggested by warning.
- `state_dict` key mismatch with `model.` prefix: loader now unwraps legacy prefixed keys.
- Eval scene sequence repeats: omit `--seed` or set different seed.
- Visualization crash with forced scene ids: choose a valid layout/style pair.
- Push rejected due large checkpoint file: remove checkpoint from history or keep it out of git.

## Quick Command Block (Copy/Paste)

```bash
source .venv/bin/activate
cd /Users/anshularavind/cs188-default-project/cabinet_door_project

python 00_verify_installation.py
python 04_download_dataset.py
python 05b_augment_handle_data.py
python 06c_train_temporal_unet_bc_policy.py --config configs/temporal_unet_bc_old_exact_400_policy3.yaml

python 07_evaluate_policy.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --num_rollouts 20 \
  --split pretrain

python 08_visualize_policy_rollout.py \
  --checkpoint /tmp/cabinet_temporal_unet_bc_ckpts/best_policy_3.pt \
  --num_episodes 1 \
  --max_steps 500
```
