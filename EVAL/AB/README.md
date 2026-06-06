# EVAL/AB/ — M4: Alpha-Beta eval + final-model thesis gate

The thesis claim lives here: **the trained RL agent beats Catanatron's
Alpha-Beta with statistical significance — win rate > 25% over ≥1000 four-player
games, 95% CI** (`PLAN.md` M4, root). 0.25 is the 4-player chance baseline.

The agent plays through `EVAL/bridge/CatanatronBridge` *inside Catanatron's reference
engine*, so the numbers are directly comparable to Catanatron paper baselines.

## Native AlphaBeta — a fast, faithful training opponent

The eval above runs Catanatron's **real** AlphaBeta through the bridge (~6.4 s/game,
crashes on P2P trades) — fine for the final gate, far too slow to *train* against.
So the same player is ported natively into the fastcatan C++ engine
(`src/catan/search.cpp`, `include/search.hpp`), exposed as:

```python
env = fastcatan.Env(); env.reset(seed)
env.ab_decide(pov, depth=2, prune=False)   # -> best flat action id (0xFFFFFFFF if none)
env.ab_value(pov)                          # -> Catanatron base_fn heuristic value
```

It is a faithful port of `catanatron.players.minimax.AlphaBetaPlayer`
(depth-2 expectimax over dice / dev-draw / robber-steal chance nodes, alpha-beta,
`list_prunned_actions`) + `value.base_fn` (`DEFAULT_WEIGHTS`). The engine refactor
that makes the chance forks possible is `rules.cpp::expand_action`
(forced-outcome cores split out of the RNG handlers; the RNG sim path stays
byte-identical — perft hash unchanged).

**Fidelity (validated, `test_native_ab_fidelity.py`, run via the bridge):**
- `ab_value` == `base_fn(DEFAULT_WEIGHTS)` to **machine precision** (worst rel
  error 1.9e-16 over 4800 state×seat pairs; exact in MAIN phase).
- On deterministic 1:1-action decisions, Catanatron's depth-1 pick achieves
  **exactly** fastcatan's best value (100 %) — every raw move difference is a
  pure value tie (different tie-break order).
- Two deliberate, documented deviations (both *more* correct than the
  reference): BUY_DEV forks the true remaining deck; robber-steal forks the
  victim's real hand (Catanatron uses an info-set blur / flat 1/5).

**Train against it** (`models/env.py`, `models/train_ppo.py`):

```bash
python -m models.train_ppo --opponent alphabeta --ab-depth 1 --num-envs 768 ...
```

Throughput (single env): `random` ~51k learner-steps/s, **depth-1 ~45k**
(≈ Catanatron `ValueFunctionPlayer`, nearly free and already crushes a random
learner), depth-2 ~5k (~10× slower). Depth-1 is the recommended training
opponent; bump to depth-2 / `--ab-prune` for a stronger curriculum. This is the
"opponent-in-pool" lever for the M4-blocked-on-M3 gap.

Pure-engine checks live in `tests/test_alphabeta.py`; the catanatron-fidelity
gate in `test_native_ab_fidelity.py` (this dir).

## Files

| File | Role |
|---|---|
| `policy.py` | wraps a trained checkpoint as a bridge `PolicyFn` (`obs, mask, rng -> int`). Registry mirrors `models/eval.py`; only `ppo` wired today. Raises on obs/action-dim mismatch. |
| `mcts_policy.py` | **state-aware** bridge policy: the bridge stashes the live `Game` each `decide()`; this injects it into a fastcatan `Env` (`bridge/state_inject`), calls `recompute_mask()` (injected states carry a stale cached mask — without this every root step is a masked no-op), runs the hybrid `MCTSvsFixed` (learned prior + `ab_value` leaves), and answers within the bridge's action mask (fallbacks counted in the result JSON). |
| `tournament.py` | the harness: policy-via-bridge vs `AlphaBetaPlayer`/`ValueFunctionPlayer`/`RandomPlayer`. Win rate + 95% Wilson CI + thesis gate → `results/*.json`. `--policy mcts` for search agents, `--rotate-seats` for seat-balanced runs (default is RED/seat-0 only), `--model-ab-depth/--model-ab-prune` to match the in-tree opponent model to the actual table. |
| `soak.py` | 10⁸-step stability soak (pure fastcatan): finite-obs + mask-integrity + leak checks. |
| `REPRODUCIBILITY.md` | toolchain, build flags, the **two-env** setup, **catanatron git pin**, seeds, train config. |
| `results/` | tournament result JSONs + `validation_1084.md` (pipeline validation). |

## Environment

The RL interface is **obs 1084 / actions 286**. The repo `.venv` carries
fastcatan + the **pinned catanatron** (3.3.0 @ git `41ba0db`, not PyPI —
newer builds move `models.tiles` → `models.map` and break the bridge import;
see `REPRODUCIBILITY.md`) and is what the current results were produced
under. `soak.py` needs only fastcatan.

```bash
# Smoke (seconds): any policy vs random through the bridge.
PYTHONPATH=.:EVAL python -m AB.tournament --games 20 --opponent random --ckpt <ckpt>

# 10^8 soak (~minutes at ~70k steps/s).
PYTHONPATH=.:EVAL python -m AB.soak --steps 100000000 --seed 7
```

The evaluated model is a `--ckpt` flag — reactive checkpoints (`--policy
ppo`, SB3 .zip) and search checkpoints (`--policy mcts`, AZ .pt) both work;
the interface must be 1084/286.

## State-aware hybrid search through the bridge (`--policy mcts`)

Reactive policies topped out at 0/200 here; the configuration that reached
parity on the native engine is a *search* agent, which needs the live game
state — wired 2026-06-06 (run under the repo `.venv`, which also carries the
pinned catanatron):

```bash
PYTHONPATH=.:EVAL python -m AB.tournament --policy mcts \
    --ckpt models/checkpoints/il_ab_d2_vpm/il_final.pt \
    --games 200 --mcts-sims 512 --model-ab-depth 2 --model-ab-prune \
    --opponent alphabeta --ab-depth 2 --ab-prune --no-trades --rotate-seats
```

**Reference results (2026-06-06), hybrid = IL-clone prior + `ab_value`
leaves (two-scale lexicographic squash, `--ab-value-scale 86e6`):**

| arena | result |
|---|---|
| native AB-d1, ≥512 sims (600 g) | **29.0% [25.5–32.8] — above 25% parity** |
| native AB-d2, 256–512 sims (600 g) | **23.3–23.75% — at parity** (was 0/200 for every reactive policy) |
| native AB-d2 *pruned* control (40 g) | 17.5% [8.8–32.0] — pruning ≈ no strength change |
| **bridge** AB-d2 pruned, rotated, 256 sims (100 g) | **5.0% [2.2–11.2]** — first consistent bridge wins ever, but 4–5× below native |

**The open native→bridge transfer gap.** Ruled out by experiment: injected
value fidelity (`test_native_ab_fidelity.py` passes to machine precision on
current code), opponent pruning strength, in-tree model depth. Live
suspects: catanatron-AB behavioral divergence from the native in-tree model
(the two documented chance-handling deviations + tie-break order — the
search optimizes against a slightly wrong opponent), sub-prompt decision
routing through the codec (robber-victim picks), and a bridge-only
sims-inversion (512 < 256: 0/40 vs 5/100) consistent with deeper search
exploiting model error harder. Next instrument: replay bridge games and
diff the in-tree model's predicted opponent moves against catanatron's
actual moves.

## Status (2026-06-06)

History, compressed: harness/soak/pin/pipeline validated 2026-05 (281/281
bridge tests, obs-identity 5/5, `results/validation_1084.md`,
`REPRODUCIBILITY.md`); every reactive policy — the 50M PPO seed (89.5% vs
`RandomPlayer` through this same bridge), 200M league self-play, arch-sweep
nets — scored **0/200–0/500 vs AlphaBeta** here, which is what motivated the
search campaign (root README).

- [x] **Native AB ladder beaten (2026-06-06):** hybrid search above parity vs
      d1 (29.0% [25.5–32.8]), at parity vs d2 (23.75% [19.8–28.2]) — see the
      campaign section in the root README for the design rules
      (dense value targets, neuro-symbolic leaves, sims scaling).
- [x] State-aware MCTS bridge policy + seat rotation wired (`--policy mcts`,
      `--rotate-seats`); `Env.recompute_mask()` added for injected states.
- [~] Official bridge gate: **5/100 = 5.0% [2.2–11.2]** vs AB-d2 (rotated) —
      first-ever consistent bridge wins (record was 0/200) but the
      native→bridge transfer gap (4–5×) is the open problem; suspects + ruled-out
      list above.
- [ ] Close the transfer gap (model-divergence differential debugging), then
      re-run the ≥1000-game thesis gate.
- [ ] Full 10⁸-step soak (smoke green; ~24 min full).
- [ ] Record thesis-gate result.
