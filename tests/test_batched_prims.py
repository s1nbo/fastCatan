"""Parity tests for the batched tree-search primitives (GPU-batched MCTS).

BatchedEnv.{save_snapshots, load_snapshots, reseed, step_raw,
write_obs_pov_batch, write_obs_all4, write_sigs} and the banned-mask
ab_decide overload were added so a search can drive the N slots as parallel
scratch branches. Every batched op must agree byte-for-byte with the
single-Env reference path, and step_raw must NOT auto-reset.
"""
from __future__ import annotations

import numpy as np

import fastcatan as fc

N = 8
WARM_STEPS = 60
SEED = 20260603


def _legal_ids(mask_words: np.ndarray) -> list[int]:
    ids = []
    for wi, word in enumerate(mask_words):
        w = int(word)
        base = wi * 64
        while w:
            bit = (w & -w).bit_length() - 1
            a = base + bit
            if a < fc.NUM_ACTIONS:
                ids.append(a)
            w &= w - 1
    return ids


def _p2p_banned_words() -> np.ndarray:
    """uint64[MASK_WORDS] with every p2p-trade action bit set (mirrors
    models.alphazero.mcts.p2p_trade_mask, inlined to keep this test
    torch-free)."""
    a = fc.action
    ids = (
        list(range(a.TRADE_ADD_GIVE_BASE, a.TRADE_ADD_GIVE_BASE + 5))
        + list(range(a.TRADE_ADD_WANT_BASE, a.TRADE_ADD_WANT_BASE + 5))
        + [a.TRADE_OPEN, a.TRADE_ACCEPT, a.TRADE_DECLINE]
        + list(range(a.TRADE_CONFIRM_BASE, a.TRADE_CONFIRM_BASE + 4))
        + [a.TRADE_CANCEL]
    )
    words = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    for i in ids:
        if i < fc.NUM_ACTIONS:
            words[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    return words


def _warm_batch(seed=SEED, steps=WARM_STEPS):
    """BatchedEnv with N mid-game, mutually different states (step_raw only,
    so no auto-reset can have occurred)."""
    be = fc.BatchedEnv(N, seed)
    be.reset()
    rng = np.random.default_rng(seed)
    masks = np.zeros((N, fc.MASK_WORDS), dtype=np.uint64)
    acts = np.zeros(N, dtype=np.uint32)
    rew = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.uint8)
    for _ in range(steps):
        be.write_masks(masks)
        for i in range(N):
            legal = _legal_ids(masks[i])
            acts[i] = rng.choice(legal) if legal else fc.SKIP_ACTION
        be.step_raw(acts, rew, done)
    return be


def _save(be) -> np.ndarray:
    buf = np.zeros((N, fc.SNAPSHOT_BYTES), dtype=np.uint8)
    be.save_snapshots(buf)
    return buf


def _sigs(be) -> np.ndarray:
    out = np.zeros((N, fc.SIG_INTS), dtype=np.int32)
    be.write_sigs(out)
    return out


def _env_sig_row(env: "fc.Env") -> list[int]:
    return (
        [env.current_player, env.phase, env.flag, env.dice_roll]
        + [env.player_handsize(p) for p in range(4)]
        + [env.player_vp(p) for p in range(4)]
    )


def test_save_rows_match_per_env_snapshot():
    be = _warm_batch()
    buf = _save(be)
    for i in range(N):
        assert buf[i].tobytes() == be.snapshot(i)


def test_save_load_roundtrip_through_env():
    be = _warm_batch()
    buf = _save(be)
    env = fc.Env()
    for i in range(N):
        env.load_snapshot(buf[i].tobytes())
        assert env.phase == be.phase(i)
        assert env.current_player == be.current_player(i)
        for p in range(4):
            assert env.player_vp(p) == be.player_vp(i, p)
            assert env.player_handsize(p) == be.player_handsize(i, p)
    # load_snapshots restores the exact bytes
    be2 = fc.BatchedEnv(N, 1)
    be2.reset()
    be2.load_snapshots(buf)
    assert np.array_equal(_save(be2), buf)


def test_write_sigs_matches_env_probes():
    be = _warm_batch()
    buf = _save(be)
    sigs = _sigs(be)
    env = fc.Env()
    for i in range(N):
        env.load_snapshot(buf[i].tobytes())
        assert sigs[i].tolist() == _env_sig_row(env)


def test_obs_pov_batch_and_all4_match_env():
    be = _warm_batch()
    buf = _save(be)
    all4 = np.zeros((N, 4, fc.OBS_SIZE), dtype=np.float32)
    be.write_obs_all4(all4)
    env = fc.Env()
    ref = np.zeros(fc.OBS_SIZE, dtype=np.float32)
    for i in range(N):
        env.load_snapshot(buf[i].tobytes())
        for pov in range(4):
            env.write_obs(pov, ref)
            assert np.array_equal(all4[i, pov], ref), (i, pov)
    # pov-vector variant: pick a different pov per env
    povs = np.arange(N, dtype=np.uint8) % 4
    out = np.zeros((N, fc.OBS_SIZE), dtype=np.float32)
    be.write_obs_pov_batch(povs, out)
    for i in range(N):
        assert np.array_equal(out[i], all4[i, povs[i]]), i


def test_step_raw_parity_with_env():
    """Same snapshot + same reseed + same action == identical successor,
    reward and done — including chance steps (dice/draws), which is the
    property batched MCTS descent relies on."""
    be = _warm_batch()
    env = fc.Env()
    rng = np.random.default_rng(99)
    masks = np.zeros((N, fc.MASK_WORDS), dtype=np.uint64)
    acts = np.zeros(N, dtype=np.uint32)
    rew = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.uint8)
    mask1 = np.zeros(fc.MASK_WORDS, dtype=np.uint64)

    for _round in range(5):
        buf = _save(be)
        seeds = rng.integers(0, 2**63, size=N, dtype=np.uint64)
        be.write_masks(masks)
        for i in range(N):
            legal = _legal_ids(masks[i])
            acts[i] = rng.choice(legal) if legal else fc.SKIP_ACTION
        be.reseed(seeds)
        be.step_raw(acts, rew, done)
        post = _save(be)
        post_sigs = _sigs(be)

        for i in range(N):
            env.load_snapshot(buf[i].tobytes())
            if acts[i] == fc.SKIP_ACTION:
                assert post[i].tobytes() == buf[i].tobytes()
                continue
            env.reseed(int(seeds[i]))
            r, d = env.step(int(acts[i]))
            assert r == rew[i], i
            assert int(d) == int(done[i]), i
            assert env.snapshot() == post[i].tobytes(), i
            assert post_sigs[i].tolist() == _env_sig_row(env), i


def test_step_raw_skip_leaves_env_untouched():
    be = _warm_batch()
    before = _save(be)
    acts = np.full(N, fc.SKIP_ACTION, dtype=np.uint32)
    rew = np.ones(N, dtype=np.float32)
    done = np.ones(N, dtype=np.uint8)
    be.step_raw(acts, rew, done)
    assert np.array_equal(_save(be), before)
    assert not rew.any() and not done.any()  # outputs cleared for skipped envs


def test_step_raw_no_autoreset_on_done():
    """Drive random games with step_raw until one terminates: the terminal
    state must persist (normal `step` would auto-reset to a fresh
    INITIAL_PLACEMENT_1 game)."""
    be = fc.BatchedEnv(N, SEED + 1)
    be.reset()
    rng = np.random.default_rng(SEED + 1)
    masks = np.zeros((N, fc.MASK_WORDS), dtype=np.uint64)
    acts = np.zeros(N, dtype=np.uint32)
    rew = np.zeros(N, dtype=np.float32)
    done = np.zeros(N, dtype=np.uint8)
    finished = -1
    for _ in range(60000):
        be.write_masks(masks)
        for i in range(N):
            legal = _legal_ids(masks[i])
            acts[i] = rng.choice(legal) if legal else fc.SKIP_ACTION
        be.step_raw(acts, rew, done)
        if done.any():
            finished = int(np.nonzero(done)[0][0])
            break
    assert finished >= 0, "no game terminated within the step budget"
    # Terminal state persists: not a fresh initial-placement reset...
    assert be.phase(finished) != 0
    sig_before = _sigs(be)[finished].copy()
    # ...and SKIP keeps it byte-stable.
    acts[:] = fc.SKIP_ACTION
    be.step_raw(acts, rew, done)
    assert np.array_equal(_sigs(be)[finished], sig_before)


def test_ab_decide_banned_mask():
    be = _warm_batch()
    buf = _save(be)
    sigs = _sigs(be)
    banned = _p2p_banned_words()
    zeros = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    banned_ids = set(_legal_ids(banned))
    env = fc.Env()
    mask1 = np.zeros(fc.MASK_WORDS, dtype=np.uint64)
    checked = 0
    for i in range(N):
        env.load_snapshot(buf[i].tobytes())
        pov = int(sigs[i][0])
        env.action_mask(mask1)
        legal = _legal_ids(mask1)
        if len(legal) <= 1:
            continue
        plain = env.ab_decide(pov, 1, False)
        with_zeros = env.ab_decide(pov, 1, False, zeros)
        assert plain == with_zeros, "all-zeros mask must equal no mask"
        pick = env.ab_decide(pov, 1, False, banned)
        assert pick in legal
        assert pick not in banned_ids, (
            "banned-mask pick must stay inside the caller's filtered set"
        )
        # never-strand: banning EVERYTHING falls back to the unfiltered set
        all_banned = np.full(fc.MASK_WORDS, np.uint64(0xFFFFFFFFFFFFFFFF),
                             dtype=np.uint64)
        pick_all = env.ab_decide(pov, 1, False, all_banned)
        assert pick_all in legal
        checked += 1
    assert checked > 0, "warm states produced no multi-legal decision points"
