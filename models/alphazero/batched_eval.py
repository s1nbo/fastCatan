"""Batched evaluation helpers for the GPU-batched AlphaZero stack.

v1: raw-policy (no search) vs random, fully batched — the cheap learning-curve
probe the batched trainer logs periodically. Seat 0 plays argmax of the masked
policy logits; seats 1-3 play uniform-random legal. E games run in lockstep on
one BatchedEnv via step_raw (finished games park on SKIP), so a 128-game eval
costs a few seconds instead of minutes.

Search-based batched eval (BatchedMCTS at high sims, AB opponents via the
banned-mask ab_decide) is the M4 follow-up and will extend this module.
"""
from __future__ import annotations

import numpy as np
import torch

import fastcatan

from models.alphazero.mcts import p2p_trade_mask

OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS
MASK_WORDS = fastcatan.MASK_WORDS
NUM_PLAYERS = fastcatan.NUM_PLAYERS
SIG_INTS = fastcatan.SIG_INTS
SKIP = fastcatan.SKIP_ACTION
WIN_VP = 10

_SHIFTS = np.arange(64, dtype=np.uint64)


def _unpack(words: np.ndarray) -> np.ndarray:
    bits = (words[:, :, None] >> _SHIFTS[None, None, :]) & np.uint64(1)
    return bits.reshape(words.shape[0], -1)[:, :NUM_ACTIONS].astype(bool)


@torch.no_grad()
def eval_vs_random_raw(
    net: torch.nn.Module,
    device: str,
    games: int = 128,
    seed: int = 0,
    suppress_p2p: bool = True,
    max_rounds: int = 60000,
) -> tuple[float, int]:
    """Raw-policy win rate at seat 0 vs 3 random seats.

    Returns (win_rate_over_decided_games, no_winner_count). Deterministic
    given (net, seed).
    """
    E = games
    env = fastcatan.BatchedEnv(E, seed)
    env.reset()
    rng = np.random.default_rng(seed ^ 0xE7A1)
    p2p = p2p_trade_mask() if suppress_p2p else None

    masks_u64 = np.zeros((E, MASK_WORDS), dtype=np.uint64)
    sigs = np.zeros((E, SIG_INTS), dtype=np.int32)
    obs = np.zeros((E, OBS_SIZE), dtype=np.float32)
    acts = np.zeros(E, dtype=np.uint32)
    rew = np.zeros(E, dtype=np.float32)
    done = np.zeros(E, dtype=np.uint8)
    finished = np.zeros(E, dtype=bool)
    winners = np.full(E, -1, dtype=np.int8)

    net.eval()
    for _ in range(max_rounds):
        if finished.all():
            break
        env.write_masks(masks_u64)
        env.write_sigs(sigs)
        legal = _unpack(masks_u64)
        if p2p is not None:
            f = legal & ~p2p[None, :]
            empty = ~f.any(axis=1)
            if empty.any():
                f[empty] = legal[empty]
            legal = f

        live = np.nonzero(~finished)[0]
        seat0 = live[sigs[live, 0] == 0]
        acts[:] = SKIP
        if seat0.size:
            env.write_obs(obs)              # current-player POV rows
            t = torch.from_numpy(obs[seat0]).to(device)
            logits, _ = net(t)
            lg = logits.float().cpu().numpy()
            lg[~legal[seat0]] = -np.inf
            acts[seat0] = lg.argmax(axis=1).astype(np.uint32)
        for g in live:
            if sigs[g, 0] != 0:
                ids = np.nonzero(legal[g])[0]
                if ids.size:
                    acts[g] = np.uint32(rng.choice(ids))

        env.step_raw(acts, rew, done)
        if done.any():
            env.write_sigs(sigs)
            for g in np.nonzero(done)[0]:
                if finished[g]:
                    continue
                finished[g] = True
                vps = sigs[g, 8:12]
                for p in range(NUM_PLAYERS):
                    if vps[p] >= WIN_VP:
                        winners[g] = p
                        break

    decided = int((winners >= 0).sum())
    wins = int((winners == 0).sum())
    no_winner = E - decided
    return (wins / decided if decided else 0.0), no_winner
