#!/usr/bin/env python3
"""Differential comparison: random self-play in fastCatan vs Catanatron.

Both simulators play games with random legal actions; we collect aggregate
stats and print side-by-side. This is the statistical-comparison version of
the differential test (PLAN.md M1). Per-step byte-exact replay (with action
ID translation) lands in M2.

Usage:
    source .venv/bin/activate
    bash tools/build_lib.sh
    python3 tools/compare_random_play.py [--games N]

Notes:
    * fastCatan side uses ctypes shim, picks uniformly random from compute_mask.
    * Catanatron side uses RandomPlayer, plays full game.
    * Catanatron has more dev cards (M1 only ships knight); some skew expected.
"""

from __future__ import annotations
import argparse
import ctypes
import random
import re
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# fastCatan ctypes wrapper
# ---------------------------------------------------------------------
SETTLE_BASE, CITY_BASE, ROAD_BASE = 0, 54, 108
ROLL_DICE, END_TURN = 180, 181
DISCARD_BASE, MOVE_ROBBER_BASE, STEAL_BASE = 182, 187, 206
TRADE_BASE = 210
BUY_DEV, PLAY_KNIGHT = 235, 236
NUM_ACTIONS = 237
MASK_WORDS = 5

PHASE_ENDED = 3

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "build" / ("libfastcatan.dylib" if sys.platform == "darwin" else "libfastcatan.so")
if not LIB.exists():
    print(f"missing {LIB}; run `bash tools/build_lib.sh`")
    sys.exit(1)
lib = ctypes.CDLL(str(LIB))

VP, U8, U16, U64, U32, I = (ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int)
def _b(n, restype, *argtypes):
    f = getattr(lib, n); f.restype = restype; f.argtypes = list(argtypes); return f

create  = _b("fcatan_create",  VP)
destroy = _b("fcatan_destroy", None, VP)
reset_  = _b("fcatan_reset",   None, VP, U64)
step_   = _b("fcatan_step",    U8,   VP, U32)
phase   = _b("fcatan_phase",   U8,   VP)
turn_count   = _b("fcatan_turn_count",       U16, VP)
start_player = _b("fcatan_start_player",     U8,  VP)
p_vp     = _b("fcatan_player_vp",     U8, VP, I)
p_hand   = _b("fcatan_player_handsize", U8, VP, I)
bank     = _b("fcatan_bank",          U8, VP, I)
p_res    = _b("fcatan_player_resource", U8, VP, I, I)
compute_mask = _b("fcatan_compute_mask", None, VP, ctypes.POINTER(U64))


def play_one_fastcatan(env, seed, rng):
    """Play one game with random legal actions. Returns stats dict."""
    reset_(env, seed)
    arr = (U64 * MASK_WORDS)()
    steps = 0
    safety = 50_000
    while phase(env) != PHASE_ENDED and safety > 0:
        safety -= 1
        compute_mask(env, arr)
        # collect set bits
        bits = []
        for w in range(MASK_WORDS):
            v = arr[w]
            while v:
                lsb = v & (-v)
                idx = lsb.bit_length() - 1
                bits.append(w * 64 + idx)
                v ^= lsb
        if not bits:
            break
        a = rng.choice(bits)
        step_(env, a)
        steps += 1

    sp = start_player(env)
    vps = [p_vp(env, p) for p in range(4)]
    winner = next((p for p in range(4) if vps[p] >= 10), -1)

    bsum = sum(bank(env, r) for r in range(5))
    hsum = sum(p_res(env, p, r) for p in range(4) for r in range(5))

    return dict(
        steps=steps,
        turns=turn_count(env),
        winner=winner,
        winner_relative=(winner - sp) % 4 if winner >= 0 else -1,
        final_vps=vps,
        terminated=(phase(env) == PHASE_ENDED),
        resource_total=bsum + hsum,
    )


def play_n_fastcatan(n_games, base_seed=42):
    rng = random.Random(0)
    env = create()
    stats = []
    t0 = time.perf_counter()
    for i in range(n_games):
        stats.append(play_one_fastcatan(env, base_seed + i, rng))
    elapsed = time.perf_counter() - t0
    destroy(env)
    return stats, elapsed


# ---------------------------------------------------------------------
# Catanatron side
# ---------------------------------------------------------------------
def play_n_catanatron(n_games, seed=42):
    from catanatron import Game, Color, RandomPlayer
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    stats = []
    t0 = time.perf_counter()
    for i in range(n_games):
        g = Game([RandomPlayer(c) for c in colors], seed=seed + i)
        g.play()

        ps = g.state.player_state
        # Catanatron may permute seat order; map via state.color_to_index.
        c2i = g.state.color_to_index
        vps = [0]*4
        for c, idx in c2i.items():
            vps[idx] = ps[f'P{idx}_ACTUAL_VICTORY_POINTS']

        winner_color = g.winning_color()
        winner = c2i[winner_color] if winner_color is not None else -1
        start_color = g.state.actions[0].color if g.state.actions else None
        start_idx = c2i[start_color] if start_color is not None else 0

        stats.append(dict(
            steps=len(g.state.actions),
            turns=g.state.num_turns,
            winner=winner,
            winner_relative=(winner - start_idx) % 4 if winner >= 0 else -1,
            final_vps=vps,
            terminated=(winner >= 0),
        ))
    elapsed = time.perf_counter() - t0
    return stats, elapsed


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def summarize(name, stats, elapsed):
    n = len(stats)
    finished = [s for s in stats if s["terminated"]]
    n_fin = len(finished)
    print(f"  games:         {n} ({n_fin} terminated, {(n_fin/n*100):.1f}%)")
    print(f"  elapsed:       {elapsed:.2f} s ({n/elapsed:.1f} games/s)")
    if finished:
        steps = [s["steps"] for s in finished]
        turns = [s["turns"] for s in finished]
        print(f"  steps/game:    mean={statistics.mean(steps):.0f}  median={statistics.median(steps):.0f}  min={min(steps)}  max={max(steps)}")
        print(f"  turns/game:    mean={statistics.mean(turns):.0f}  median={statistics.median(turns):.0f}")
        # winner distribution by relative seat
        from collections import Counter
        rel = Counter(s["winner_relative"] for s in finished if s["winner_relative"] >= 0)
        total = sum(rel.values())
        if total:
            rel_str = ", ".join(f"seat+{k}={rel[k]/total*100:.1f}%" for k in sorted(rel))
            print(f"  win by seat:   {rel_str}")
        # final vp avg
        all_vps = [v for s in finished for v in s["final_vps"]]
        print(f"  final VP mean: {statistics.mean(all_vps):.2f}  (max VP across players in finished games)")
        # winner vp = should always be 10
        winner_vps = [s["final_vps"][s["winner"]] for s in finished]
        print(f"  winner VP:     mean={statistics.mean(winner_vps):.2f}  min={min(winner_vps)}  max={max(winner_vps)}")
    if "resource_total" in stats[0]:
        bad = [s for s in stats if s["resource_total"] != 95]
        print(f"  resource conservation violations: {len(bad)} / {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    args = ap.parse_args()

    print(f"Running {args.games} random-play games in each simulator...")
    print()
    print("=== fastCatan ===")
    fc_stats, fc_t = play_n_fastcatan(args.games)
    summarize("fastCatan", fc_stats, fc_t)
    print()
    print("=== Catanatron ===")
    ct_stats, ct_t = play_n_catanatron(args.games)
    summarize("Catanatron", ct_stats, ct_t)
    print()
    print("=== Comparison ===")
    fc_fin = [s for s in fc_stats if s["terminated"]]
    ct_fin = [s for s in ct_stats if s["terminated"]]
    if fc_fin and ct_fin:
        fc_steps = statistics.mean(s["steps"] for s in fc_fin)
        ct_steps = statistics.mean(s["steps"] for s in ct_fin)
        fc_turns = statistics.mean(s["turns"] for s in fc_fin)
        ct_turns = statistics.mean(s["turns"] for s in ct_fin)
        print(f"  steps/game ratio: fastCatan/Catanatron = {fc_steps/ct_steps:.2f}")
        print(f"  turns/game ratio: fastCatan/Catanatron = {fc_turns/ct_turns:.2f}")
    print(f"  speed:            fastCatan = {args.games/fc_t:.0f} g/s   Catanatron = {args.games/ct_t:.1f} g/s   ({fc_t and (ct_t/fc_t):.1f}x)")


if __name__ == "__main__":
    main()
