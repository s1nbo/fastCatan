#!/usr/bin/env python3
"""Smoke tests for the tournament harness."""
from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


def test_basic_play():
    a = fc.random_legal_policy_for_eval(rng=np.random.default_rng(0))
    b = fc.random_legal_policy_for_eval(rng=np.random.default_rng(1))
    result = fc.play(agent_a=a, agent_b=b, n_games=10, seed=42,
                      num_envs=4, max_steps_per_game=20000)
    fails = 0
    fails += fail(result.n_games == 10, f"n_games = {result.n_games}")
    fails += fail(result.wins_a + result.wins_b + result.ties + result.truncated == 10,
                  f"games sum: {result.wins_a + result.wins_b + result.ties + result.truncated}")
    fails += fail(0.0 <= result.win_rate_a <= 1.0, f"win_rate_a = {result.win_rate_a}")
    fails += fail(result.ci95_a[0] <= result.win_rate_a <= result.ci95_a[1],
                  f"CI doesn't bracket win rate: {result.ci95_a} vs {result.win_rate_a}")
    print(f"  {result.n_games} games: A={result.wins_a} B={result.wins_b} "
          f"trunc={result.truncated} avg_len={result.avg_game_length:.0f}")
    return fails


def test_lowest_vs_random():
    """Deterministic policy vs random — at least the games must terminate."""
    a = fc.lowest_legal_policy_for_eval()
    b = fc.random_legal_policy_for_eval(rng=np.random.default_rng(7))
    result = fc.play(agent_a=a, agent_b=b, n_games=10, seed=99,
                      num_envs=4, max_steps_per_game=30000)
    fails = 0
    completed = result.wins_a + result.wins_b + result.ties
    fails += fail(completed >= 1, f"no games completed: trunc={result.truncated}")
    print(f"  lowest vs random: A={result.wins_a} B={result.wins_b} "
          f"trunc={result.truncated}")
    return fails


def test_seeded_reproducibility():
    """Same seed + same policies should yield the same aggregate stats."""
    a1 = fc.lowest_legal_policy_for_eval()
    b1 = fc.lowest_legal_policy_for_eval()
    r1 = fc.play(agent_a=a1, agent_b=b1, n_games=8, seed=42,
                  num_envs=4, max_steps_per_game=20000)
    a2 = fc.lowest_legal_policy_for_eval()
    b2 = fc.lowest_legal_policy_for_eval()
    r2 = fc.play(agent_a=a2, agent_b=b2, n_games=8, seed=42,
                  num_envs=4, max_steps_per_game=20000)
    fails = 0
    fails += fail(r1.wins_a == r2.wins_a, f"wins_a: {r1.wins_a} vs {r2.wins_a}")
    fails += fail(r1.wins_b == r2.wins_b, f"wins_b: {r1.wins_b} vs {r2.wins_b}")
    fails += fail(r1.truncated == r2.truncated, f"trunc: {r1.truncated} vs {r2.truncated}")
    return fails


def test_seat_plan_override():
    """All-A vs all-B: A should win 100% in completed games."""
    a = fc.random_legal_policy_for_eval(rng=np.random.default_rng(0))
    b = fc.random_legal_policy_for_eval(rng=np.random.default_rng(1))
    seats = [["a", "a", "a", "a"]] * 5  # 4 A-seats per game; A wins all
    result = fc.play(agent_a=a, agent_b=b, n_games=5, seed=1,
                      num_envs=4, max_steps_per_game=20000,
                      seats_per_game=seats)
    fails = 0
    completed = result.wins_a + result.wins_b + result.ties
    fails += fail(completed > 0, "no games completed")
    fails += fail(result.wins_b == 0, f"B won when only A seats: wins_b={result.wins_b}")
    return fails


def main():
    total = 0
    print("== test_basic_play =="); total += test_basic_play()
    print("== test_lowest_vs_random =="); total += test_lowest_vs_random()
    print("== test_seeded_reproducibility =="); total += test_seeded_reproducibility()
    print("== test_seat_plan_override =="); total += test_seat_plan_override()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
