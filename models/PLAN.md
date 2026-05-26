# models/ — RL Training Plan

> Status: scaffolding landed. `env.py`, `train_ppo.py`, `train_a2c.py`,
> `train_dqn.py`, `train_muzero.py`, `eval.py` all present. Final
> checkpoints stashed under `models/checkpoints/{a2c,dqn,muzero,ppo_random}/`.
> Remaining: **gate run** — record ≥1000-game eval of PPO checkpoint vs
> random with win rate ≥0.90 (M2 thesis gate). Obs/reward freeze before
> M3 self-play.

## Context

Project enters **M2** (PLAN.md:193). Need first RL agent. Targets:
- Industry-standard algos, each in own file, all simple to read. Coverage:
  Q-Learning (DQN), Actor-Critic (A2C), PPO (MaskablePPO via sb3-contrib),
  MuZero (model-based + MCTS).
- All share one Gymnasium env (`env.py`) over `fastcatan.Env` (724-dim obs,
  296 discrete actions, 320-bit legal-action mask).
- Learner controls seat 0; seats 1–3 = uniform-random-legal opponents
  (M2 gate: >90% vs random).
- Keep scripts *simple* — each file self-contained, no premature abstraction.
  Self-play (M3) and AlphaBeta eval (M4) come later in their own scripts.

## Layout

```
models/
├── PLAN.md              (this file)
├── env.py               Gymnasium env wrapping single fastcatan.Env (POV=seat0, opponents=random)
├── train_ppo.py         PPO (SB3 MaskablePPO)        — clipped surrogate, baseline
├── train_a2c.py         A2C (custom, ~200 lines)     — simplest actor-critic w/ mask
├── train_dqn.py         DQN/Q-Learning (custom)      — Q-net + target net + replay + eps-greedy
├── train_muzero.py      MuZero scaffold (custom)     — repr+dyn+pred nets + MCTS, demonstrative
├── eval.py              Load PPO checkpoint, play N games vs random, win rate + 95% CI
└── checkpoints/         (gitignored) snapshots + tensorboard logs
```

No package `__init__.py`. Scripts run via `python -m models.train_<algo>` or direct.

### Algo overview (intentionally simple)

| File | Style | Lib | When to read |
|------|-------|-----|--------------|
| `train_dqn.py` | value-based, off-policy | pure torch | learn DQN from scratch |
| `train_a2c.py` | policy-gradient + baseline, on-policy | pure torch | bridge from REINFORCE to PPO |
| `train_ppo.py` | clipped policy-gradient, on-policy | sb3-contrib | M2 gate run |
| `train_muzero.py` | model-based, MCTS planning | pure torch | reference for model-based RL |

Each file is self-contained — no shared utility module — so they can be read top-to-bottom in isolation.

## Design decisions

### 1. VecEnv adapter, not single-env

`BatchedEnv` is already vectorized (N envs, contiguous buffers, zero-copy numpy). Wrap it as `sb3_contrib`-compatible `VecEnv` directly — do **not** wrap one-at-a-time and then re-vectorize via `SubprocVecEnv` (would throw away the C++ vectorization).

`env.py` exposes:
- `FastCatanVecEnv(num_envs, seed, opponent_policy="random") -> VecEnv`
- Implements `reset()`, `step_async()/step_wait()`, `action_masks()` (required by MaskablePPO).
- Observation space: `Box(low=-inf, high=inf, shape=(OBS_SIZE,), dtype=float32)`.
- Action space: `Discrete(NUM_ACTIONS)` (296).
- **Inner loop**: after learner steps for seat 0, advance C++ env until `current_player()==0` again by sampling opponent actions from `opponent_policy(obs, mask)`. Random uses uniform-over-legal from `examples/player_base.py:legal_actions`.

### 2. Reward = native sim signal

Use as-is: +1 on win-action, -1 if action lets opponent win, 0 else (bindings.cpp). No shaping in M2. Locking obs+reward is itself an M2 deliverable (`PLAN.md:196`).

### 3. Action masking

`MaskablePPO.predict(obs, action_masks=...)` from `sb3_contrib`. Pull mask via `BatchedEnv.write_masks(buf)` → unpack uint64[5] → bool[296] per env. Helper already exists in `examples/player_base.py:legal_actions` — reuse.

### 4. Training defaults (`train_ppo.py`)

Industry defaults, light tweaks for discrete-action self-play:
- `policy="MlpPolicy"` (Box obs → no CNN needed; 724 dims, 2× 256 hidden by default)
- `n_steps=512`, `batch_size=4096`, `n_epochs=4`
- `learning_rate=3e-4`, `gamma=0.999` (Catan episodes are long, ~60–200 actions/seat)
- `ent_coef=0.01`, `clip_range=0.2`
- `num_envs=512` default (BatchedEnv handles this trivially)
- Tensorboard logging to `models/checkpoints/tb/`
- `CheckpointCallback` every 500k steps

### 5. Eval (`eval.py`)

- Load `.zip` checkpoint.
- Spin a small `BatchedEnv` (N=64), run 1000 games seat-0=learner vs seats-1..3=random.
- Compute win rate, 95% Wilson CI. Threshold: 0.90.
- Reuses `examples/player_base.legal_actions` for opponent action sampling.

## Files to add/touch

| File | New? | Purpose |
|------|------|---------|
| `models/PLAN.md` | new | this doc |
| `models/env.py` | new | `FastCatanVecEnv` |
| `models/train_ppo.py` | new | MaskablePPO trainer |
| `models/eval.py` | new | win-rate eval vs random |
| `pyproject.toml` | edit | add `[project.optional-dependencies] rl = ["torch", "stable-baselines3>=2.3", "sb3-contrib>=2.3", "gymnasium>=0.29", "tensorboard"]` |
| `.gitignore` | edit | add `models/checkpoints/` |

## Reused code (no new abstractions)

- `python/fastcatan/__init__.py` — `BatchedEnv`, `OBS_SIZE`, `MASK_WORDS`, `NUM_ACTIONS`
- `examples/player_base.py:legal_actions` — bit-unpack mask
- `examples/random_player.py` — opponent policy template
- `bridge/run_eval.py` — eval-loop shape (for later M4 cross-check vs Catanatron)

## Verification

1. **Smoke**: `python -m models.train_ppo --num-envs 64 --total-steps 100_000` runs without crash, tensorboard shows non-zero `ep_rew_mean` after ~30s.
2. **Mask correctness**: assert in `env.step_wait()` that `mask[action] == 1` for each picked action (debug flag).
3. **Determinism**: same seed → same first-1000-step obs/reward stream.
4. **Gate check**: after ≥5M steps, `python -m models.eval --ckpt latest --games 1000` reports win rate >0.90 vs random. M2 gate met.
5. **Throughput**: log steps/sec; should land within 2× of pure `BatchedEnv.step` benchmark (Risk 3 in root PLAN.md).
