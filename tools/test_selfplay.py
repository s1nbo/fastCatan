#!/usr/bin/env python3
"""Smoke tests for selfplay policy adapters."""
from __future__ import annotations
import sys
import numpy as np

import fastcatan as fc


def fail(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 1
    return 0


class _MockModel:
    """Mimics SB3.predict — picks the first legal action from the bool mask."""
    def predict(self, obs, action_masks=None, deterministic=True):
        idx = int(np.argmax(action_masks))
        return np.array([idx], dtype=np.int64), None


def test_policy_from_sb3_signature():
    model = _MockModel()
    policy = fc.policy_from_sb3(model, deterministic=True)
    # Call with the tournament Policy signature.
    obs = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    mask_packed = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    mask_packed[0] = 1 << 5  # action 5 legal
    a = policy(obs, mask_packed, 0, 0, None)
    return fail(int(a) == 5, f"expected 5, got {a}")


def test_frozen_self_play_opponent_signature():
    model = _MockModel()
    opp = fc.FrozenSelfPlayOpponent(model, deterministic=True)
    obs = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    mask_packed = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    mask_packed[0] = 1 << 7
    a = opp(obs, mask_packed)
    return fail(int(a) == 7, f"expected 7, got {a}")


def test_play_via_sb3_wrapper():
    """Tournament harness accepts sb3-wrapped policies."""
    a = fc.policy_from_sb3(_MockModel(), deterministic=True)
    b = fc.random_legal_policy_for_eval(rng=np.random.default_rng(0))
    result = fc.play(agent_a=a, agent_b=b, n_games=4, seed=42,
                      num_envs=2, max_steps_per_game=15000)
    return fail(result.n_games == 4, "tournament didn't run")


def main():
    total = 0
    print("== test_policy_from_sb3_signature ==");        total += test_policy_from_sb3_signature()
    print("== test_frozen_self_play_opponent_signature =="); total += test_frozen_self_play_opponent_signature()
    print("== test_play_via_sb3_wrapper ==");             total += test_play_via_sb3_wrapper()
    print()
    if total == 0: print("ALL TESTS PASS")
    else:          print(f"FAIL — {total} failures")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
