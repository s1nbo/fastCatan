# fastCatan

High-throughput simulator for 4-player Settlers of Catan, built for
reinforcement-learning research. C++23 core, nanobind bindings, batched
stepping with optional OpenMP, Gymnasium + PettingZoo wrappers.

> **Project context.** Companion simulator for an RL thesis on Catan.
> Goal: train an RL agent that beats classical Alpha-Beta with
> statistical significance over ≥1000 four-player games. The bottleneck
> the thesis explicitly calls out is simulator throughput — existing
> Python sims (Catanatron) can't generate self-play at the volume modern
> RL needs.

## Throughput

| Configuration | Steps/sec |
|---|---|
| Pure C++ batched (1024 envs, 1 thread) | ~13M |
| Python (nanobind, step + mask) | ~19M |
| HPC with OpenMP across 32 cores | projected ~150-300M |

For comparison, M1 PLAN target was 5×10⁵; we're ~25× over.

## Quickstart

### Local (macOS / Linux)

```bash
git clone <repo> fastcatan
cd fastcatan
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install cmake ninja nanobind scikit-build-core    # build deps
pip install -e . --no-build-isolation                  # builds the C++ extension
pip install torch sb3-contrib stable-baselines3         # for training
```

Verify:

```bash
python3 -c "import fastcatan as fc; print(fc.OBS_SIZE, fc.NUM_ACTIONS)"
# 724 296
```

### HPC

See [`HPC.md`](HPC.md) — module loads, GCC 14.2 + CUDA setup, SLURM job template.

## Hello world

### Single-env, raw API

```python
import fastcatan as fc
import numpy as np

env = fc.Env()
env.reset(seed=42)
print(f"phase={env.phase}, current_player={env.current_player}")

# Pick a legal action and step.
reward, done = env.step(fc.action.SETTLE_BASE + 5)   # build settlement at node 5
```

### Batched env (the hot path)

```python
import fastcatan as fc
import numpy as np

n = 4096
env = fc.BatchedEnv(num_envs=n, seed=42)
env.reset()

# Pre-allocate buffers; nanobind passes them through to C++ zero-copy.
actions = np.zeros(n, dtype=np.uint32)
rewards = np.zeros(n, dtype=np.float32)
dones   = np.zeros(n, dtype=np.uint8)
masks   = np.zeros((n, fc.MASK_WORDS), dtype=np.uint64)
obs     = np.zeros((n, fc.OBS_SIZE), dtype=np.float32)

env.write_masks(masks)
# ... your policy fills `actions` ...
env.step(actions, rewards, dones)
env.write_obs(obs)
```

### Gymnasium wrapper (single-agent vs random opponents)

```python
import fastcatan as fc
import numpy as np

rng = np.random.default_rng(0)
opp = fc.random_legal_policy(rng)
env = fc.GymEnv(seat=0, seed=42, opponent_fn=opp)

obs, info = env.reset(seed=42)
mask = info["action_mask"]              # bool[NUM_ACTIONS] for SB3
packed = info["action_mask_packed"]      # uint64[MASK_WORDS] raw

action = my_policy(obs, mask)
obs, reward, terminated, truncated, info = env.step(action)
```

### PettingZoo AEC (multi-agent / trading-net training)

```python
import fastcatan as fc

env = fc.CatanAECEnv(seed=42)
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None); continue
    action = my_policies[agent](obs["observation"], obs["action_mask"])
    env.step(action)
```

### Training with MaskablePPO

```bash
python3 tools/train_smoke.py --total-timesteps 10000 --device auto
```

See [`tools/train_smoke.py`](tools/train_smoke.py) for the full setup.

## Key concepts

### Action space

Flat `Discrete(NUM_ACTIONS=296)`:

```
0..53     SETTLE   build settlement at node N
54..107   CITY     build city at node N
108..179  ROAD     build road at edge E
180       ROLL_DICE
181       END_TURN
182..186  DISCARD  by resource (during DISCARD sub-phase)
187..205  MOVE_ROBBER  to hex H
206..209  STEAL    from player P
210..234  TRADE    bank/port: give*5 + get
235       BUY_DEV
236       PLAY_KNIGHT
237       PLAY_ROAD_BUILDING
238..262  PLAY_YEAR_OF_PLENTY  (give1*5 + give2)
263..267  PLAY_MONOPOLY  by resource
268..272  TRADE_ADD_GIVE_BASE
273..277  TRADE_REMOVE_GIVE_BASE
278..282  TRADE_ADD_WANT_BASE
283..287  TRADE_REMOVE_WANT_BASE
288       TRADE_OPEN
289       TRADE_ACCEPT
290       TRADE_DECLINE
291..294  TRADE_CONFIRM_BASE  by partner
295       TRADE_CANCEL
```

All action IDs are exposed as `fastcatan.action.<NAME>`.

### Action mask

Mask is a `uint64[5]` (320 bits, 296 used). Bit `i` set ⇔ action `i` is
legal. Read via `env.write_masks(buf)` (BatchedEnv) or
`info["action_mask_packed"]` (GymEnv). For SB3-style consumers, GymEnv
also unpacks to a `bool[NUM_ACTIONS]` in `info["action_mask"]`.

The mask is incrementally maintained inside `GameState` and refreshed on
every `step_one`. Sub-action types (trade compose) use a surgical updater
for ~21% throughput win.

### Observation

`obs` is a `float32[OBS_SIZE=724]` from the current player's perspective.
Fields are POV-relative (self always at slot 0, opponents at +1, +2, +3).

### Reward

`+1` to the actor on the action that pushes their VP to 10. `-1` to the
actor if their action somehow triggered another player's win (rare).
Everything else is `0`. Loser perspective is reconstructed in the AEC
wrapper / training loop.

### RNG

xoshiro128++ per env, 16 bytes of state, seeded via SplitMix64 from a
master seed. Deterministic given the seed; perft hashes pinned in
`tools/perft_hashes.json` are guarded by CI.

## Repo layout

```
include/                 public headers (state, rules, mask, obs, batched_env, rng, topology)
src/catan/               core C++ implementation
bindings/pycatan/        nanobind module
python/fastcatan/        Python package (re-exports + Gym/PettingZoo wrappers)
tools/                   build scripts, tests, benchmarks, training smoke
bench/                   pure-C++ throughput benchmarks
CMakeLists.txt           build system (HPC-ready)
pyproject.toml           scikit-build-core editable install
PLAN.md                  thesis plan + milestone tracking
HPC.md                   HPC build + SLURM setup
```

## Tests

16 test suites covering rules, mask consistency, observation encoding,
batched env, longest-road corpus, PvP trade, perft determinism, and the
nanobind / Gym / PettingZoo wrappers.

```bash
for t in tools/test_*.py; do echo "=== $t ==="; python3 "$t" 2>&1 | tail -1; done
```

## Status

- ✅ All Catan rules implemented (initial placement → end game)
- ✅ Largest army + longest road with strict-exceed transfer rules
- ✅ All 5 dev cards (knight, VP, road-building, year-of-plenty, monopoly)
- ✅ Player-to-player trading (compose → respond → confirm)
- ✅ Batched env + auto-reset + OpenMP-ready
- ✅ Gymnasium + PettingZoo wrappers
- ✅ MaskablePPO training pipeline
- ✅ HPC build setup
- ✅ Tournament harness (`fc.play(a, b, n_games)` with Wilson 95% CIs)
- ✅ Alpha-Beta baseline (depth-limited minimax with multi-feature heuristic, ~75% vs random at depth 2)
- ✅ Self-play wrappers (`policy_from_sb3`, `FrozenSelfPlayOpponent`)

See [`PLAN.md`](PLAN.md) for the full milestone roadmap.

## License

MIT — see [`LICENSE`](LICENSE).
