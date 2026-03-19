"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

This script supports both:
- New diffusion U-Net checkpoints from 06_train_policy.py
- Staged BC checkpoints from 06b_train_staged_bc_policy.py
- Legacy MLP checkpoints

Success criterion used here: opening any one target cabinet door.
"""

import argparse
import os
import sys

from runtime_setup import configure_offscreen_gl, select_torch_device

configure_offscreen_gl()

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env

from policy_common import is_one_door_open_success, load_policy_wrapper


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_evaluation(
    policy,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
    success_threshold,
):
    """Run evaluation rollouts and collect statistics."""
    import imageio

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        policy.reset()

        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            action = policy.act(obs=obs, env=env, env_action_dim=env.action_dim)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if is_one_door_open_success(env, threshold=success_threshold):
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.10,
        help="Door openness threshold for one-door success",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = select_torch_device()
    print(f"Device: {device}")

    policy, info, ckpt = load_policy_wrapper(args.checkpoint, device)
    print(f"Loaded policy from: {args.checkpoint}")
    print(f"  Type:      {info.model_type}")
    print(f"  Epoch:     {info.epoch}")
    print(f"  Loss:      {info.loss:.6f}")
    print(f"  State dim: {info.state_dim}, Action dim: {info.action_dim}")

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")
    print(f"Success rule: any one door open (threshold={args.success_threshold:.2f})")

    results = run_evaluation(
        policy=policy,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        success_threshold=args.success_threshold,
    )

    print_section("Evaluation Results")

    num_success = int(sum(results["successes"]))
    success_rate = num_success / args.num_rollouts * 100
    avg_length = float(np.mean(results["episode_lengths"]))
    avg_reward = float(np.mean(results["rewards"]))

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")


if __name__ == "__main__":
    main()
