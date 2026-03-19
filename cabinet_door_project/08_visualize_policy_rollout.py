"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint and runs it in OpenCabinet so you can watch
what the policy is doing.

Supports:
- Diffusion U-Net checkpoints from 06_train_policy.py
- Staged BC checkpoints from 06b_train_staged_bc_policy.py
- Legacy MLP checkpoints

Success criterion used here: opening any one target cabinet door.
"""

import os
import sys

from runtime_setup import configure_offscreen_gl, select_torch_device

# Configure rendering backend before other heavy imports.
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    configure_offscreen_gl()
else:
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"

import argparse
import time

import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from policy_common import (
    OnlineHandleFeatureExtractor,
    is_one_door_open_success,
    load_policy_wrapper,
)


def _collect_debug_metrics(extractor, env, action, near_handle_threshold):
    feat = extractor.extract(env)

    handle_to_eef = feat.get("handle_to_eef_pos", np.zeros((3,), dtype=np.float32))
    h = np.asarray(handle_to_eef, dtype=np.float32).reshape(-1)
    if h.shape[0] >= 3:
        dist_to_handle = float(np.linalg.norm(h[:3]))
    else:
        dist_to_handle = float("inf")

    door_open_arr = feat.get("door_openness", np.zeros((1,), dtype=np.float32))
    door_open_flat = np.asarray(door_open_arr, dtype=np.float32).reshape(-1)
    door_openness = float(door_open_flat[0]) if door_open_flat.size > 0 else 0.0

    gripper_cmd = float(action[6]) if action.shape[0] > 6 else float("nan")
    near_handle = bool(np.isfinite(dist_to_handle) and dist_to_handle <= near_handle_threshold)
    gripper_close = bool(gripper_cmd > 0.5)
    gripper_open = bool(gripper_cmd < -0.5)

    return {
        "dist_to_handle": dist_to_handle,
        "door_openness": door_openness,
        "gripper_cmd": gripper_cmd,
        "near_handle": near_handle,
        "gripper_close": gripper_close,
        "gripper_open": gripper_open,
    }


def run_onscreen(policy, args):
    """Run policy with interactive viewer window."""
    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    extractor = OnlineHandleFeatureExtractor()
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        policy.reset()

        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")

        success = False
        min_dist = float("inf")
        max_open = 0.0
        close_cmd_count = 0
        open_cmd_count = 0
        near_close_cmd_count = 0
        near_steps = 0

        for step in range(args.max_steps):
            action = policy.act(obs=obs, env=env, env_action_dim=env.action_dim)
            obs, reward, done, info = env.step(action)

            metrics = None
            if args.debug_metrics:
                metrics = _collect_debug_metrics(
                    extractor=extractor,
                    env=env,
                    action=action,
                    near_handle_threshold=args.near_handle_threshold,
                )
                min_dist = min(min_dist, metrics["dist_to_handle"])
                max_open = max(max_open, metrics["door_openness"])
                if metrics["gripper_close"]:
                    close_cmd_count += 1
                if metrics["gripper_open"]:
                    open_cmd_count += 1
                if metrics["near_handle"]:
                    near_steps += 1
                    if metrics["gripper_close"]:
                        near_close_cmd_count += 1

            if step % 20 == 0:
                checking = is_one_door_open_success(env, threshold=args.success_threshold)
                status = "one door OPEN" if checking else "in progress"
                if args.debug_metrics:
                    print(
                        f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                        f"  dist={metrics['dist_to_handle']:.3f}"
                        f"  open={metrics['door_openness']:.3f}"
                        f"  grip={metrics['gripper_cmd']:+.1f}"
                    )
                else:
                    print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

            if args.debug_metrics and step % max(1, int(args.debug_every)) == 0:
                print(
                    f"    debug t={step:4d}"
                    f"  dist={metrics['dist_to_handle']:.3f}"
                    f"  open={metrics['door_openness']:.3f}"
                    f"  grip={metrics['gripper_cmd']:+.1f}"
                    f"  near={int(metrics['near_handle'])}"
                )

            if is_one_door_open_success(env, threshold=args.success_threshold):
                success = True
                break

            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open a door"
        print(f"\n  Result: {result}")
        if args.debug_metrics:
            print("  Debug summary:")
            print(f"    min dist to handle:   {min_dist:.4f}")
            print(f"    max door openness:    {max_open:.4f}")
            print(f"    close cmd count:      {close_cmd_count}")
            print(f"    open cmd count:       {open_cmd_count}")
            print(f"    near-handle steps:    {near_steps}")
            print(f"    near close cmd count: {near_close_cmd_count}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


def run_offscreen(policy, args):
    """Run policy headlessly and save rollout video."""
    import imageio
    from robocasa.utils.env_utils import create_env

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768
    successes = 0
    all_frames = []
    extractor = OnlineHandleFeatureExtractor()

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )

        obs = env.reset()
        policy.reset()

        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        success = False
        ep_frames = []
        min_dist = float("inf")
        max_open = 0.0
        close_cmd_count = 0
        open_cmd_count = 0
        near_close_cmd_count = 0
        near_steps = 0

        for step in range(args.max_steps):
            action = policy.act(obs=obs, env=env, env_action_dim=env.action_dim)
            obs, reward, done, info = env.step(action)

            metrics = None
            if args.debug_metrics:
                metrics = _collect_debug_metrics(
                    extractor=extractor,
                    env=env,
                    action=action,
                    near_handle_threshold=args.near_handle_threshold,
                )
                min_dist = min(min_dist, metrics["dist_to_handle"])
                max_open = max(max_open, metrics["door_openness"])
                if metrics["gripper_close"]:
                    close_cmd_count += 1
                if metrics["gripper_open"]:
                    open_cmd_count += 1
                if metrics["near_handle"]:
                    near_steps += 1
                    if metrics["gripper_close"]:
                        near_close_cmd_count += 1

            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            if step % 20 == 0:
                checking = is_one_door_open_success(env, threshold=args.success_threshold)
                status = "one door OPEN" if checking else "in progress"
                if args.debug_metrics:
                    print(
                        f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                        f"  dist={metrics['dist_to_handle']:.3f}"
                        f"  open={metrics['door_openness']:.3f}"
                        f"  grip={metrics['gripper_cmd']:+.1f}"
                    )
                else:
                    print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

            if args.debug_metrics and step % max(1, int(args.debug_every)) == 0:
                print(
                    f"    debug t={step:4d}"
                    f"  dist={metrics['dist_to_handle']:.3f}"
                    f"  open={metrics['door_openness']:.3f}"
                    f"  grip={metrics['gripper_cmd']:+.1f}"
                    f"  near={int(metrics['near_handle'])}"
                )

            if is_one_door_open_success(env, threshold=args.success_threshold):
                success = True
                break

        result = "SUCCESS" if success else "did not open a door"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if args.debug_metrics:
            print("  Debug summary:")
            print(f"    min dist to handle:   {min_dist:.4f}")
            print(f"    max door openness:    {max_open:.4f}")
            print(f"    close cmd count:      {close_cmd_count}")
            print(f"    open cmd count:       {open_cmd_count}")
            print(f"    near-handle steps:    {near_steps}")
            print(f"    near close cmd count: {near_close_cmd_count}")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for saved video",
    )
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment layout/style selection",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.10,
        help="Door openness threshold for one-door success",
    )
    parser.add_argument(
        "--debug_metrics",
        action="store_true",
        help="Log handle distance, openness, and gripper command diagnostics",
    )
    parser.add_argument(
        "--debug_every",
        type=int,
        default=20,
        help="How often (steps) to print detailed debug metrics",
    )
    parser.add_argument(
        "--near_handle_threshold",
        type=float,
        default=0.06,
        help="Distance threshold (m) for near-handle debug stats",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with: python 06_train_policy.py")
        sys.exit(1)

    device = select_torch_device()
    policy, info, ckpt = load_policy_wrapper(args.checkpoint, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Type:      {info.model_type}")
    print(f"  Epoch:     {info.epoch}")
    print(f"  Loss:      {info.loss:.6f}")
    print(f"  State dim: {info.state_dim}, Action dim: {info.action_dim}")
    print(f"  Device:    {device}")
    print(f"  Success:   any one door open (threshold={args.success_threshold:.2f})")
    if args.debug_metrics:
        print(
            "  Debug:     enabled "
            f"(every={args.debug_every} steps, near={args.near_handle_threshold:.3f}m)"
        )

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:       {mode}")
    print(f"Episodes:   {args.num_episodes}")
    print(f"Max steps:  {args.max_steps}")
    if args.offscreen:
        print(f"Output:     {args.video_path}")

    if args.offscreen:
        run_offscreen(policy, args)
    else:
        print("Opening viewer window...")
        run_onscreen(policy, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
