"""Invariant tests for BatchedMCTS and the batched self-play driver.

CPU-only (device='cpu') so the suite runs anywhere; the GPU path is the same
code with a different torch device. These guard the lockstep search semantics:
policy targets are proper distributions over legal actions, the search is
deterministic given its seed, inactive rows pass through untouched, and the
driver's lockstep game loop records/finishes games coherently.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import fastcatan as fc

from models.alphazero.net import PolicyValueNet
from models.alphazero.batched_mcts import BatchedMCTS, SNAP


def _fresh_roots(G: int, seed: int = 99) -> np.ndarray:
    genv = fc.BatchedEnv(G, seed)
    genv.reset()
    roots = np.zeros((G, SNAP), dtype=np.uint8)
    genv.save_snapshots(roots)
    return roots


def _net(seed: int = 0) -> PolicyValueNet:
    torch.manual_seed(seed)
    net = PolicyValueNet(hidden=(64, 64))
    net.eval()
    return net


def test_search_policy_invariants():
    G, sims = 8, 16
    bm = BatchedMCTS(_net(), G, device="cpu", sims=sims, seed=1)
    roots = _fresh_roots(G)
    pi, mask, tm = bm.search(roots)
    for g in range(G):
        assert abs(pi[g].sum() - 1.0) < 1e-5
        assert (pi[g][~mask[g]] == 0).all()
        assert mask[g].any()
        assert 0 <= tm[g] <= 3
    acts = bm.choose(pi, mask, np.ones(G))
    for g in range(G):
        assert mask[g][acts[g]]


def test_search_deterministic_given_seed():
    G, sims = 4, 12
    roots = _fresh_roots(G)
    net = _net()
    pi1, _, _ = BatchedMCTS(net, G, device="cpu", sims=sims, seed=5).search(roots)
    pi2, _, _ = BatchedMCTS(net, G, device="cpu", sims=sims, seed=5).search(roots)
    assert np.array_equal(pi1, pi2)


def test_search_active_subset():
    G, sims = 6, 8
    bm = BatchedMCTS(_net(), G, device="cpu", sims=sims, seed=2)
    roots = _fresh_roots(G)
    active = np.zeros(G, dtype=bool)
    active[::2] = True
    pi, mask, tm = bm.search(roots, active=active)
    for g in range(G):
        if active[g]:
            assert abs(pi[g].sum() - 1.0) < 1e-5
            assert tm[g] >= 0
        else:
            assert pi[g].sum() == 0
            assert not mask[g].any()
            assert tm[g] == -1


def test_driver_records_and_steps():
    from models.alphazero.batched_selfplay import BatchedSelfplay

    args = argparse.Namespace(
        num_games=8, sims=12, c_puct=1.5, seed=3, allow_trades=False,
        device="cpu", buffer_size=10000, batch_size=64, value_coef=1.0,
        value_mode="sparse",
    )
    sp = BatchedSelfplay(_net(), args)
    for _ in range(12):
        sp.move_step(temp_moves=20)
    # every move-step records exactly one decision per game
    assert sp.decisions_done == 12 * args.num_games
    live_records = sum(len(s.records) for s in sp.slots)
    buffered = len(sp.buffer)
    assert live_records + buffered == sp.decisions_done
    for s in sp.slots:
        for obs, pi, mask, seat in s.records[:2]:
            assert obs.shape == (fc.OBS_SIZE,)
            assert abs(pi.sum() - 1.0) < 1e-5
            assert (pi[~mask] == 0).all()
            assert 0 <= seat <= 3
    # finished games (if any) produced +-1 z targets in the buffer
    for obs, pi, mask, z in list(sp.buffer)[:8]:
        assert z in (-1.0, 1.0)
