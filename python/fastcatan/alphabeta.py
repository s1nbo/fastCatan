"""Depth-limited alpha-beta minimax player for fastCatan.

Inspired by Catanatron's ``AlphaBetaPlayer``. Performs a forward search from
the current state, branching on every legal action up to ``depth`` plies,
and evaluates leaves with a tunable heuristic (default: VP-weighted
multi-feature score).

Stochastic actions (dice roll, robber steal, dev-card draw) are stepped
deterministically through the env's seeded RNG — the search effectively
sees ONE rollout of the random outcomes per branch. This is a deliberate
simplification (Catanatron does the same); a true expectimax would need
to enumerate dice / steal outcomes.

Usage::

    import fastcatan as fc
    from fastcatan.alphabeta import AlphaBetaPlayer

    ab = AlphaBetaPlayer(depth=2)
    result = fc.play(
        agent_a=ab,
        agent_b=fc.random_legal_policy_for_eval(),
        n_games=50, seed=42, num_envs=8,
    )

The signature matches :data:`fastcatan.tournament.Policy` so it plugs
directly into ``play``.
"""
from __future__ import annotations
import math
from typing import Callable, Optional

import numpy as np

import fastcatan as fc


# Type for evaluation function. Takes a snapshotted env (single Env) and
# the seat we're rooting for; returns a score (higher is better for that seat).
EvalFn = Callable[["fc.Env", int], float]


# Heuristic weights. Tuned to reward building over hoarding resources.
W_VP = 200.0           # total VP — biggest signal
W_PUBLIC_VP = 20.0
W_SETTLE_ON_BOARD = 30.0   # 5 - settlement_count_remaining
W_CITY_ON_BOARD = 60.0     # 4 - city_count_remaining (bonus over the +1 VP city already gives)
W_ROAD_ON_BOARD = 4.0      # 15 - road_count_remaining
W_KNIGHTS = 10.0
W_ROAD_LEN = 6.0
W_PORTS = 5.0
W_LR_TITLE = 50.0
W_LA_TITLE = 50.0

WIN_SCORE = 10_000.0


def _popcount(x: int) -> int:
    return bin(int(x)).count("1")


def default_heuristic(env: "fc.Env", learner_seat: int) -> float:
    """Multi-feature heuristic. Higher = better for ``learner_seat``.

    Rewards pieces on board (not just VP) so depth-1 search picks build
    actions over end-turn even when the post-state VP delta is zero.
    """
    score = 0.0
    for seat in range(fc.NUM_PLAYERS):
        sign = +1.0 if seat == learner_seat else -1.0 / 3.0
        score += sign * W_VP * env.player_vp(seat)
        score += sign * W_PUBLIC_VP * env.player_vp_public(seat)
        # Pieces actually on the board (caps - remaining-in-stock).
        settles_built = 5 - env.player_settlement_count(seat)
        cities_built = 4 - env.player_city_count(seat)
        roads_built = 15 - env.player_road_count(seat)
        score += sign * (W_SETTLE_ON_BOARD * settles_built
                          + W_CITY_ON_BOARD * cities_built
                          + W_ROAD_ON_BOARD * roads_built)
        score += sign * W_KNIGHTS * env.player_knights_played(seat)
        score += sign * W_ROAD_LEN * env.player_road_length(seat)
        score += sign * W_PORTS * _popcount(env.player_ports(seat))

    if env.longest_road_owner == learner_seat:
        score += W_LR_TITLE
    if env.largest_army_owner == learner_seat:
        score += W_LA_TITLE

    return score


def terminal_score(env: "fc.Env", learner_seat: int) -> Optional[float]:
    """Return +/- WIN_SCORE if the game is over, else None."""
    if env.phase != 3:  # Phase::ENDED == 3
        return None
    for seat in range(fc.NUM_PLAYERS):
        if env.player_vp(seat) >= 10:
            return WIN_SCORE if seat == learner_seat else -WIN_SCORE
    return 0.0  # shouldn't happen — terminal but no winner


def _legal_actions(mask_words: np.ndarray) -> list[int]:
    bits: list[int] = []
    for w in range(fc.MASK_WORDS):
        v = int(mask_words[w])
        base = w * 64
        while v:
            lsb = v & (-v)
            bits.append(base + (lsb.bit_length() - 1))
            v ^= lsb
    return bits


# Compose action ID range — TRADE_ADD/REMOVE GIVE/WANT increments. These
# don't change resources, just trade scratch. Search rarely benefits from
# expanding them (they balloon the branching factor without changing
# game-state value). Default AB pruning excludes them.
_COMPOSE_LO = 268   # TRADE_ADD_GIVE_BASE
_COMPOSE_HI = 288   # exclusive — TRADE_OPEN onward stays in the search.


def _drop_compose(actions: list[int]) -> list[int]:
    return [a for a in actions if a < _COMPOSE_LO or a >= _COMPOSE_HI]


class AlphaBetaPlayer:
    """Depth-limited alpha-beta search policy.

    Args:
        depth:    plies of search (1-3 is practical; deeper gets very slow).
        eval_fn:  heuristic over a single ``Env`` for a given seat. Defaults to
                  :func:`default_heuristic`.
        action_limit: optional cap on actions explored per node (sorted by
                  early heuristic; trims branching factor when there are many
                  trade-compose actions). 0 disables limiting.
    """

    def __init__(
        self,
        depth: int = 2,
        eval_fn: Optional[EvalFn] = None,
        action_limit: int = 0,
        prune_compose: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.eval_fn = eval_fn or default_heuristic
        self.action_limit = action_limit
        self.prune_compose = prune_compose
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._scratch = fc.Env()
        self._mask_buf = np.zeros(fc.MASK_WORDS, dtype=np.uint64)

    def __call__(
        self,
        obs: np.ndarray,
        mask_packed: np.ndarray,
        env_idx: int,
        seat: int,
        env: "fc.BatchedEnv",
    ) -> int:
        # Take a snapshot of the live env's slot.
        snap = env.snapshot(env_idx)

        # Run search rooted at this position.
        best_action, _ = self._search(
            snap=snap,
            depth=self.depth,
            alpha=-math.inf,
            beta=math.inf,
            learner_seat=seat,
        )
        return int(best_action)

    def _search(
        self,
        snap: bytes,
        depth: int,
        alpha: float,
        beta: float,
        learner_seat: int,
    ) -> tuple[int, float]:
        """Returns (best_action, best_value) from the snapshotted state."""
        self._scratch.load_snapshot(snap)

        # Terminal check.
        ts = terminal_score(self._scratch, learner_seat)
        if ts is not None:
            return 0, ts

        # Depth cutoff.
        if depth == 0:
            return 0, self.eval_fn(self._scratch, learner_seat)

        # Enumerate legal actions.
        self._scratch.action_mask(self._mask_buf)
        legal = _legal_actions(self._mask_buf)
        if not legal:
            return 0, self.eval_fn(self._scratch, learner_seat)

        # Compose-action pruning — keep TRADE_OPEN onward, drop the
        # ADD/REMOVE GIVE/WANT increments which balloon branching factor.
        # If compose-only legal actions remain (no other moves), keep them.
        if self.prune_compose:
            pruned = _drop_compose(legal)
            if pruned:
                legal = pruned

        if self.action_limit > 0 and len(legal) > self.action_limit:
            legal = self._top_actions(snap, legal, learner_seat, self.action_limit)

        active_seat = self._scratch.current_player
        maximizing = (active_seat == learner_seat)

        # Random shuffle so equal-scored actions break ties uniformly across
        # games (avoids systematic bias toward low-ID actions like "settle node 0").
        legal = list(legal)
        self._rng.shuffle(legal)

        best_action = legal[0]
        best_value = -math.inf if maximizing else math.inf

        for a in legal:
            self._scratch.load_snapshot(snap)
            self._scratch.step(a)
            child_snap = self._scratch.snapshot()
            _, val = self._search(child_snap, depth - 1, alpha, beta, learner_seat)

            if maximizing:
                if val > best_value:
                    best_value = val
                    best_action = a
                if best_value > alpha:
                    alpha = best_value
            else:
                if val < best_value:
                    best_value = val
                    best_action = a
                if best_value < beta:
                    beta = best_value
            if beta <= alpha:
                break

        return best_action, best_value

    def _top_actions(
        self,
        snap: bytes,
        legal: list[int],
        learner_seat: int,
        k: int,
    ) -> list[int]:
        """Score each legal action via a 1-step lookahead and keep top-k.

        Used to prune the branching factor when, e.g., trade-compose
        actions explode the local mask. Increases search depth at the
        cost of some single-ply rollouts.
        """
        scored: list[tuple[float, int]] = []
        for a in legal:
            self._scratch.load_snapshot(snap)
            self._scratch.step(a)
            scored.append((self.eval_fn(self._scratch, learner_seat), a))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [a for _, a in scored[:k]]
