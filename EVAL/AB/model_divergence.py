"""Model-divergence differential: in-tree native AB vs catanatron's real AB.

The native→bridge transfer gap (hybrid search: 23.75% vs native AB-d2, 5.0%
through the bridge) has one prime suspect: the search plans against the
NATIVE AB port as its in-tree opponent model, but the actual table plays
catanatron's AlphaBetaPlayer — which differs by documented design (BUY_DEV
info-set blur vs true-deck fork, flat-1/5 robber steal vs true-hand fork,
tie-break order, prune list). If the model mispredicts the real opponent,
the search optimizes lines the table never plays.

This harness plays N games of 4 × catanatron AB (the exact table the gate
uses), intercepts every multi-action decision via Game.play_tick's
decide_fn, mirrors the state into fastcatan (state_inject + recompute_mask),
and asks the native model what IT would play for that seat. Reports
agreement by action type / game phase, and for disagreements the native
1-ply value delta between the two choices, bucketed lexicographically:
  tie      delta == 0           (pure tie-break divergence — harmless-ish)
  sub-VP   0 < delta < 3e14     (fine-feature preference differences)
  VP-level delta >= 3e14        (the model thinks catanatron blundered a VP
                                 — these are where search exploits a phantom)

Run:
    PYTHONPATH=.:EVAL python -m AB.model_divergence --games 40 \
        --ab-depth 2 --ab-prune --model-depth 2 --model-prune
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from catanatron import Color
from catanatron.game import Game

import fastcatan
from bridge.state_inject import inject
from bridge.action_codec import encode_to_fast_ids
from models.alphazero.mcts import p2p_banned_words

from AB.tournament import TradeSafeAlphaBetaPlayer, COLORS

_NO_ACTION = 0xFFFFFFFF
VP_W = 3e14

# action ids whose execution is stochastic — skip the 1-ply value delta for
# them (stepping the mirror would compare different chance draws).
_a = fastcatan.action
_STOCHASTIC = {_a.ROLL_DICE, _a.BUY_DEV} | set(
    range(_a.STEAL_BASE, _a.STEAL_BASE + 4))


def _delta_bucket(delta: float) -> str:
    d = abs(delta)
    if d == 0.0:
        return "tie"
    if d < VP_W:
        return "sub_vp"
    return "vp_level"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--ab-depth", type=int, default=2,
                   help="catanatron table players' depth")
    p.add_argument("--ab-prune", action="store_true")
    p.add_argument("--model-depth", type=int, default=2,
                   help="native in-tree model depth to test")
    p.add_argument("--model-prune", action="store_true")
    p.add_argument("--model-catanatron-chance", action="store_true",
                   help="model uses Catanatron's chance blur (flat-1/5 "
                        "steals + info-set BUY_DEV) instead of true forks")
    p.add_argument("--out", type=str, default="AB/results")
    args = p.parse_args()

    env = fastcatan.Env()
    env.reset(0)
    banned = p2p_banned_words()

    stats = defaultdict(lambda: [0, 0])          # key -> [decisions, agree]
    delta_buckets = defaultdict(int)             # bucket -> count (disagreements)
    delta_sum_subvp = []
    unmapped = 0
    total = agree_total = 0
    turn_stats = defaultdict(lambda: [0, 0])     # turn-bucket -> [n, agree]

    def divergence_decide(player, game, playable_actions):
        nonlocal unmapped, total, agree_total
        action = player.decide(game, playable_actions)
        if len(playable_actions) <= 1:
            return action

        seat = game.state.color_to_index[player.color]
        inject(env, game)
        env.recompute_mask()
        native = env.ab_decide(seat, args.model_depth, args.model_prune,
                               banned,
                               1 if args.model_catanatron_chance else 0)
        try:
            fids = encode_to_fast_ids(action)
        except (KeyError, ValueError):
            fids = []
        if not fids:
            unmapped += 1
            return action

        ok = native != _NO_ACTION and native in fids
        total += 1
        agree_total += ok
        key = action.action_type.name
        stats[key][0] += 1
        stats[key][1] += ok
        tb = min(game.state.num_turns // 20, 4)   # turn buckets of 20
        turn_stats[tb][0] += 1
        turn_stats[tb][1] += ok

        if not ok and native != _NO_ACTION:
            cat_fid = fids[0]
            if native not in _STOCHASTIC and cat_fid not in _STOCHASTIC:
                # 1-ply native value delta between the two choices.
                snap = env.snapshot()
                env.reseed(12345)
                env.step(int(native))
                v_native = env.ab_value(seat)
                env.load_snapshot(snap)
                env.reseed(12345)
                env.step(int(cat_fid))
                v_cat = env.ab_value(seat)
                env.load_snapshot(snap)
                delta = v_native - v_cat
                b = _delta_bucket(delta)
                delta_buckets[b] += 1
                if b == "sub_vp":
                    delta_sum_subvp.append(delta)
            else:
                delta_buckets["stochastic_skip"] += 1
        return action

    t0 = time.time()
    winners = []
    for g in range(args.games):
        seed = args.seed + g
        players = [TradeSafeAlphaBetaPlayer(c, depth=args.ab_depth,
                                            prunning=args.ab_prune)
                   for c in COLORS]
        game = Game(players, seed=seed)
        random.seed(seed)
        while game.winning_color() is None and game.state.num_turns < 1000:
            game.play_tick(decide_fn=divergence_decide)
        winners.append(str(game.winning_color()))
        if (g + 1) % 5 == 0:
            rate = agree_total / max(total, 1)
            print(f"[{g+1}/{args.games}] decisions={total} "
                  f"agree={rate:.3f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== model divergence: native(d={args.model_depth},"
          f"prune={args.model_prune}) vs catanatron(d={args.ab_depth},"
          f"prune={args.ab_prune}) ===")
    print(f"decisions: {total}  agreement: {agree_total/max(total,1):.4f}  "
          f"unmapped: {unmapped}")
    print("\nby action type:")
    for k, (n, a) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:<22} n={n:>5}  agree={a/n:.3f}")
    print("\nby turn bucket (x20 turns):")
    for tb in sorted(turn_stats):
        n, a = turn_stats[tb]
        print(f"  turns {tb*20:>3}-{tb*20+19:<3} n={n:>5}  agree={a/n:.3f}")
    print("\ndisagreement value-deltas (native 1-ply view):")
    for b in ("tie", "sub_vp", "vp_level", "stochastic_skip"):
        print(f"  {b:<16} {delta_buckets.get(b, 0)}")
    if delta_sum_subvp:
        arr = np.array(delta_sum_subvp)
        print(f"  sub_vp delta: mean={arr.mean():.3g} "
              f"p90={np.percentile(np.abs(arr), 90):.3g}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "games": args.games, "seed": args.seed,
        "table": {"depth": args.ab_depth, "prune": args.ab_prune},
        "model": {"depth": args.model_depth, "prune": args.model_prune},
        "decisions": total, "agreement": agree_total / max(total, 1),
        "unmapped": unmapped,
        "by_action_type": {k: {"n": n, "agree": a / n}
                           for k, (n, a) in stats.items()},
        "by_turn_bucket": {f"{tb*20}-{tb*20+19}": {"n": n, "agree": a / n}
                           for tb, (n, a) in turn_stats.items()},
        "delta_buckets": dict(delta_buckets),
    }
    path = out_dir / f"model_divergence_{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
