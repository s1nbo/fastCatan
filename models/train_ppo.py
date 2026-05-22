"""MaskablePPO training entrypoint.

Run:
    python -m models.train_ppo --num-envs 16 --total-steps 5_000_000

M2 gate: >90% win rate vs random over 1000 games (see models/eval.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from models.env import FastCatanEnv


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _mask_fn(env):
    return env.action_masks()


def _make_env(seed: int):
    def _thunk():
        e = FastCatanEnv(seed=seed)
        return ActionMasker(e, _mask_fn)

    return _thunk


def _build_vec_env(num_envs: int, base_seed: int, parallel: bool):
    fns = [_make_env(base_seed + i) for i in range(num_envs)]
    if parallel and num_envs > 1:
        return SubprocVecEnv(fns)
    return DummyVecEnv(fns)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=str(CKPT_DIR))
    p.add_argument("--save-freq", type=int, default=500_000)
    p.add_argument("--run-name", type=str, default="ppo_random")
    p.add_argument("--no-parallel", action="store_true",
                   help="Use DummyVecEnv (single-process) instead of SubprocVecEnv.")
    args = p.parse_args()

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb"

    env = _build_vec_env(args.num_envs, args.seed, parallel=not args.no_parallel)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        seed=args.seed,
        tensorboard_log=str(tb_dir),
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.num_envs),
        save_path=str(save_dir),
        name_prefix="ppo",
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=ckpt_cb,
        progress_bar=True,
    )

    final = save_dir / "ppo_final.zip"
    model.save(str(final))
    print(f"[train] saved final model -> {final}")


if __name__ == "__main__":
    main()
