#!/usr/bin/env python3
"""Smoke tests for AlphaBetaPlayer."""
from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_construction():
    ab = fc.AlphaBetaPlayer(depth=1)
    fails = 0
    fails += fail(ab.depth == 1, "depth")
    return fails


def test_callable_signature():
    """AB should match the tournament Policy signature."""
    ab = fc.AlphaBetaPlayer(depth=1)
    rnd = fc.random_legal_policy_for_eval(rng=np.random.default_rng(0))
    # 4-arg call (Policy minus env) used to be supported; current expects env.
    result = fc.play(agent_a=ab, agent_b=rnd, n_games=2, seed=42,
                      num_envs=2, max_steps_per_game=15000)
    return fail(result.n_games == 2, "tournament didn't run")


def test_depth1_doesnt_collapse():
    """AB depth 1 should get within sampling noise of 50% vs random."""
    ab = fc.AlphaBetaPlayer(depth=1, rng=np.random.default_rng(0))
    rnd = fc.random_legal_policy_for_eval(rng=np.random.default_rng(1))
    result = fc.play(agent_a=ab, agent_b=rnd, n_games=40, seed=42,
                      num_envs=8, max_steps_per_game=15000)
    fails = 0
    completed = result.wins_a + result.wins_b + result.ties
    fails += fail(completed >= 35, f"only {completed}/40 finished")
    fails += fail(result.win_rate_a >= 0.30,
                  f"AB depth-1 collapsed to {result.win_rate_a*100:.0f}% vs random")
    print(f"  AB(d=1): wins_a={result.wins_a}, wins_b={result.wins_b}, "
          f"win_rate_a={result.win_rate_a:.3f}")
    return fails


def test_depth2_beats_random():
    """AB depth 2 should clearly beat random."""
    ab = fc.AlphaBetaPlayer(depth=2, action_limit=12, rng=np.random.default_rng(0))
    rnd = fc.random_legal_policy_for_eval(rng=np.random.default_rng(1))
    result = fc.play(agent_a=ab, agent_b=rnd, n_games=20, seed=42,
                      num_envs=4, max_steps_per_game=15000)
    fails = 0
    completed = result.wins_a + result.wins_b + result.ties
    fails += fail(completed >= 18, f"only {completed}/20 finished")
    fails += fail(result.win_rate_a > 0.55,
                  f"AB depth-2 only at {result.win_rate_a*100:.0f}% vs random")
    print(f"  AB(d=2): wins_a={result.wins_a}, wins_b={result.wins_b}, "
          f"win_rate_a={result.win_rate_a:.3f}")
    return fails


def test_snapshot_round_trip():
    """Env snapshot/load round trip preserves state."""
    e = fc.Env()
    e.reset(seed=42)
    snap1 = e.snapshot()
    # Step a few actions.
    mask = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    e.action_mask(mask)
    bits = []
    for w in range(fc.MASK_WORDS):
        v = int(mask[w])
        while v:
            bits.append(w*64 + (v & -v).bit_length() - 1)
            v &= v - 1
    e.step(bits[0])
    e.action_mask(mask)
    bits2 = []
    for w in range(fc.MASK_WORDS):
        v = int(mask[w])
        while v:
            bits2.append(w*64 + (v & -v).bit_length() - 1)
            v &= v - 1
    e.step(bits2[0])
    # Save second snap and verify round trip
    snap2 = e.snapshot()
    e.load_snapshot(snap1)
    fails = 0
    fails += fail(e.phase == 0, "phase didn't restore")
    e.load_snapshot(snap2)
    fails += fail(e.snapshot() == snap2, "second round trip differed")
    return fails


def main():
    total = 0
    print("== test_construction ==");           total += test_construction()
    print("== test_callable_signature ==");     total += test_callable_signature()
    print("== test_snapshot_round_trip ==");    total += test_snapshot_round_trip()
    print("== test_depth1_doesnt_collapse =="); total += test_depth1_doesnt_collapse()
    print("== test_depth2_beats_random ==");    total += test_depth2_beats_random()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
