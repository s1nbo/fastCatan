#!/usr/bin/env python3
"""Profile fastCatan + training pipeline to find the slow parts.

Reports per-component timing and percentages so you can target
optimization at the actual bottleneck (rather than guessing).

Sections:

  engine    raw env ops (step / mask / obs / reset) at the C++ boundary
  gym       single-env Gymnasium wrapper overhead (opponent loop, info dict)
  ppo       full SB3 MaskablePPO training loop (rollout + update breakdown)
  cprofile  cProfile + pstats summary of a short training run

Usage::

    source .venv/bin/activate
    python3 tools/profile_train.py --section all
    python3 tools/profile_train.py --section engine
    python3 tools/profile_train.py --section ppo --total-timesteps 2048

Interpretation tips:

  - If `engine` numbers dominate → optimize C++ side (incremental mask,
    SIMD, OpenMP) or use BatchedEnv directly (no Gym wrapper).
  - If `gym` overhead dominates → the wrapper's opponent loop / info
    dict assembly is the bottleneck. Switch to BatchedEnv + custom
    rollout, or use SubprocVecEnv to amortize.
  - If `ppo update` dominates → use a smaller policy network, larger
    batch size, or move to GPU.
  - If `ppo collect` dominates → SB3's single-env loop is the
    bottleneck → consider a custom BatchedEnv-driven PPO.
"""
from __future__ import annotations
import argparse
import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

import fastcatan as fc


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@dataclass
class Timing:
    label: str
    elapsed: float
    ops: int = 0           # number of operations performed (used for ops/sec)

    def per_op_ns(self) -> float:
        if self.ops == 0:
            return float("nan")
        return self.elapsed * 1e9 / self.ops

    def ops_per_sec(self) -> float:
        if self.ops == 0:
            return float("nan")
        return self.ops / self.elapsed


def report(rows: list[Timing], total_label: str = "total"):
    total = sum(t.elapsed for t in rows)
    print(f"  {'component':<32}  {'time (s)':>10}  {'%':>6}  {'ops':>10}  {'ns/op':>10}  {'ops/sec':>14}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*14}")
    for t in rows:
        pct = (t.elapsed / total * 100) if total > 0 else 0
        ops_str = f"{t.ops:,}" if t.ops > 0 else "—"
        ns_op = f"{t.per_op_ns():.0f}" if t.ops > 0 else "—"
        ops_sec = f"{t.ops_per_sec():,.0f}" if t.ops > 0 else "—"
        print(f"  {t.label:<32}  {t.elapsed:>10.3f}  {pct:>5.1f}%  {ops_str:>10}  {ns_op:>10}  {ops_sec:>14}")
    print(f"  {'-'*32}  {'-'*10}")
    print(f"  {total_label:<32}  {total:>10.3f}")


@contextmanager
def measure(rows: list[Timing], label: str, ops: int = 0):
    t0 = time.perf_counter()
    yield
    rows.append(Timing(label, time.perf_counter() - t0, ops))


# ---------------------------------------------------------------------
# Engine — raw C++ env ops via ctypes-equivalent (nanobind) calls
# ---------------------------------------------------------------------
def profile_engine(n_envs: int = 1024, n_iters: int = 1000):
    print(f"\n=== ENGINE (BatchedEnv n={n_envs}, n_iters={n_iters}) ===")
    env = fc.BatchedEnv(num_envs=n_envs, seed=42)
    env.reset()

    actions = np.zeros(n_envs, dtype=np.uint32)
    rewards = np.zeros(n_envs, dtype=np.float32)
    dones   = np.zeros(n_envs, dtype=np.uint8)
    masks   = np.zeros((n_envs, fc.MASK_WORDS), dtype=np.uint64)
    obs     = np.zeros((n_envs, fc.OBS_SIZE), dtype=np.float32)

    # warmup
    for _ in range(20):
        env.write_masks(masks)
        env.step(actions, rewards, dones)

    rows: list[Timing] = []
    total_steps = n_envs * n_iters

    with measure(rows, "step (illegal action 0)", total_steps):
        for _ in range(n_iters):
            env.step(actions, rewards, dones)

    with measure(rows, "write_masks", total_steps):
        for _ in range(n_iters):
            env.write_masks(masks)

    with measure(rows, "write_obs", total_steps):
        for _ in range(n_iters):
            env.write_obs(obs)

    with measure(rows, "step + mask + obs", total_steps):
        for _ in range(n_iters):
            env.write_masks(masks)
            env.step(actions, rewards, dones)
            env.write_obs(obs)

    with measure(rows, "reset", n_envs * 50):
        for _ in range(50):
            env.reset()

    report(rows, total_label="engine total")


# ---------------------------------------------------------------------
# Gym — single-env wrapper overhead (incl. opponent loop)
# ---------------------------------------------------------------------
def profile_gym(n_steps: int = 2000):
    print(f"\n=== GYM (single-env wrapper, n_steps={n_steps}) ===")
    rng = np.random.default_rng(0)
    opp = fc.random_legal_policy(rng)
    env = fc.GymEnv(seat=0, seed=42, opponent_fn=opp, max_steps=20000)
    obs, info = env.reset(seed=42)

    rows: list[Timing] = []

    # Reset cost
    with measure(rows, "reset", 50):
        for _ in range(50):
            env.reset(seed=42)

    obs, info = env.reset(seed=42)
    with measure(rows, "step (incl. opponent loop)", n_steps):
        for _ in range(n_steps):
            mask_packed = info["action_mask_packed"]
            a = opp(obs, mask_packed)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                obs, info = env.reset(seed=42)

    report(rows, total_label="gym total")


# ---------------------------------------------------------------------
# PPO — full SB3 MaskablePPO loop
# ---------------------------------------------------------------------
def profile_ppo(total_timesteps: int = 1024, device: str = "auto"):
    print(f"\n=== PPO (MaskablePPO, total_timesteps={total_timesteps}) ===")
    try:
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError as e:
        print(f"  skipped: {e}")
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device: {device}")

    rng = np.random.default_rng(0)
    opp = fc.random_legal_policy(rng)

    def mask_fn(env_):
        return env_.unwrapped._make_info()["action_mask"]

    base = fc.GymEnv(seat=0, seed=42, opponent_fn=opp, max_steps=20000)
    env = ActionMasker(base, mask_fn)

    rows: list[Timing] = []

    with measure(rows, "model construction"):
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=2,
            device=device,
            verbose=0,
            seed=42,
            policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
        )

    with measure(rows, "learn (rollout + updates)", total_timesteps):
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Eval-style predict timing.
    obs = env.reset(seed=42)[0]
    info = env.unwrapped._make_info()
    n_predict = 200
    with measure(rows, "predict (deterministic)", n_predict):
        for _ in range(n_predict):
            action, _ = model.predict(obs, action_masks=info["action_mask"],
                                       deterministic=True)

    report(rows, total_label="ppo total")
    fps = total_timesteps / next(t.elapsed for t in rows
                                  if t.label.startswith("learn"))
    print(f"\n  effective fps during learn(): {fps:.0f} env-steps/sec")


# ---------------------------------------------------------------------
# cProfile — function-level breakdown of a short training run
# ---------------------------------------------------------------------
def profile_cprofile(total_timesteps: int = 1024, top: int = 30):
    print(f"\n=== cProfile (PPO learn, top {top} by cumulative time) ===")
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError as e:
        print(f"  skipped: {e}")
        return

    rng = np.random.default_rng(0)
    opp = fc.random_legal_policy(rng)

    def mask_fn(env_):
        return env_.unwrapped._make_info()["action_mask"]

    base = fc.GymEnv(seat=0, seed=42, opponent_fn=opp, max_steps=20000)
    env = ActionMasker(base, mask_fn)
    model = MaskablePPO(
        policy="MlpPolicy", env=env, learning_rate=3e-4,
        n_steps=256, batch_size=64, n_epochs=2,
        verbose=0, seed=42,
        policy_kwargs=dict(net_arch=dict(pi=[128, 64], vf=[128, 64])),
    )

    pr = cProfile.Profile()
    pr.enable()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    pr.disable()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    stats.print_stats(top)
    print(stream.getvalue())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", default="all",
                    choices=["all", "engine", "gym", "ppo", "cprofile"],
                    help="which section(s) to run")
    ap.add_argument("--n-envs", type=int, default=1024,
                    help="batch size for engine section")
    ap.add_argument("--n-iters", type=int, default=1000,
                    help="iterations for engine section")
    ap.add_argument("--gym-steps", type=int, default=2000,
                    help="steps for gym section")
    ap.add_argument("--total-timesteps", type=int, default=1024,
                    help="timesteps for ppo / cprofile sections")
    ap.add_argument("--device", default="auto",
                    help="torch device for ppo section (cpu / mps / cuda / auto)")
    ap.add_argument("--cprofile-top", type=int, default=30,
                    help="top N functions to show in cProfile output")
    args = ap.parse_args()

    if args.section in ("all", "engine"):
        profile_engine(args.n_envs, args.n_iters)
    if args.section in ("all", "gym"):
        profile_gym(args.gym_steps)
    if args.section in ("all", "ppo"):
        profile_ppo(args.total_timesteps, args.device)
    if args.section in ("cprofile",):
        profile_cprofile(args.total_timesteps, args.cprofile_top)


if __name__ == "__main__":
    main()
