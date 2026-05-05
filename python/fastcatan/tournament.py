"""Tournament harness for evaluating policies head-to-head.

A *policy* here is any callable with signature::

    policy(obs: np.ndarray, mask_packed: np.ndarray,
           env_idx: int, seat: int, env: fc.BatchedEnv) -> int

Where:
  - ``obs``         is a float32 vector of shape ``(OBS_SIZE,)`` from this
                    seat's POV (already POV-flipped — the agent always sees
                    its own slot at index 0).
  - ``mask_packed`` is a uint64 array of shape ``(MASK_WORDS,)`` with the
                    legal-action bitmask.
  - ``env_idx``     is the index in the underlying ``BatchedEnv``.
  - ``seat``        is the seat assignment for this policy (0..3).
  - ``env``         is the live ``BatchedEnv`` (used by search-based agents
                    that need to snapshot state and explore branches).

The policy must return a single legal action ID. ``random_legal_policy_for_eval``
in this module is a built-in baseline.

The standard call is :func:`play`::

    from fastcatan.tournament import play, random_legal_policy_for_eval

    # Two random agents square off across all seats.
    result = play(
        agent_a=random_legal_policy_for_eval(rng=np.random.default_rng(0)),
        agent_b=random_legal_policy_for_eval(rng=np.random.default_rng(1)),
        n_games=1000,
        seed=42,
    )
    print(f"Agent A win rate: {result.win_rate_a:.3f} (95% CI: {result.ci95_a})")

The harness uses :class:`fastcatan.BatchedEnv` under the hood and runs
``num_envs`` games in parallel. Auto-reset is handled internally.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import math

import numpy as np

import fastcatan as fc

# Type signature for tournament policies. The trailing ``env`` parameter is
# the live BatchedEnv handle — search-based policies (alpha-beta, MCTS)
# use it to snapshot state and explore branches; reflex policies ignore it.
Policy = Callable[[np.ndarray, np.ndarray, int, int, "fc.BatchedEnv"], int]


def random_legal_policy_for_eval(
    rng: Optional[np.random.Generator] = None,
) -> Policy:
    """Return a stateless random-legal-action policy with the tournament signature."""
    rng = rng if rng is not None else np.random.default_rng()

    def policy(obs: np.ndarray, mask_packed: np.ndarray,
               env_idx: int, seat: int, env=None) -> int:
        bits = []
        for w in range(fc.MASK_WORDS):
            v = int(mask_packed[w])
            base = w * 64
            while v:
                lsb = v & (-v)
                bits.append(base + (lsb.bit_length() - 1))
                v ^= lsb
        return int(rng.choice(bits)) if bits else 0

    return policy


def lowest_legal_policy_for_eval() -> Policy:
    """Return a stateless deterministic policy that always picks the lowest-id legal action."""
    def policy(obs: np.ndarray, mask_packed: np.ndarray,
               env_idx: int, seat: int, env=None) -> int:
        for w in range(fc.MASK_WORDS):
            v = int(mask_packed[w])
            if v:
                return w * 64 + (v & -v).bit_length() - 1
        return 0
    return policy


@dataclass
class TournamentResult:
    """Aggregate result of :func:`play`.

    Attributes:
        n_games:           total games played
        n_completed:       games that reached terminal state (Phase::ENDED)
        wins_a:            number of games where any agent A seat won
        wins_b:            number of games where any agent B seat won
        ties:              completed games where neither A nor B won (rare)
        truncated:         games stopped at ``max_steps`` without termination
        win_rate_a:        wins_a / n_completed
        win_rate_b:        wins_b / n_completed
        ci95_a:            95% Wilson confidence interval for A's win rate
        ci95_b:            95% Wilson confidence interval for B's win rate
        avg_game_length:   mean number of step calls per completed game
        seat_a:            list of seats assigned to agent A this tournament
        seat_b:            list of seats assigned to agent B this tournament
    """
    n_games: int
    n_completed: int
    wins_a: int
    wins_b: int
    ties: int
    truncated: int
    win_rate_a: float
    win_rate_b: float
    ci95_a: tuple[float, float]
    ci95_b: tuple[float, float]
    avg_game_length: float
    seat_a: list[int]
    seat_b: list[int]


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfw = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def _enumerate_legal(mask_packed: np.ndarray) -> list[int]:
    bits: list[int] = []
    for w in range(fc.MASK_WORDS):
        v = int(mask_packed[w])
        base = w * 64
        while v:
            lsb = v & (-v)
            bits.append(base + (lsb.bit_length() - 1))
            v ^= lsb
    return bits


def play(
    agent_a: Policy,
    agent_b: Policy,
    n_games: int,
    seed: int = 42,
    seats_per_game: Optional[Sequence[Sequence[int]]] = None,
    num_envs: Optional[int] = None,
    max_steps_per_game: int = 20_000,
    verbose: bool = False,
) -> TournamentResult:
    """Run an A-vs-B tournament and return aggregate statistics.

    Each game has 4 seats. By default agent A occupies 2 seats and agent B
    occupies 2 seats (seats [0, 2] for A, [1, 3] for B). Seats can be
    overridden per-game via ``seats_per_game`` — list of length 4 with each
    entry either 'a' or 'b'.

    Args:
        agent_a, agent_b: callables matching the :data:`Policy` signature.
        n_games:          total games to play.
        seed:             master seed (per-game seeds derived from this).
        seats_per_game:   optional per-game seat assignment override; if
                          omitted, agent A gets seats 0+2 and agent B gets
                          seats 1+3 in every game.
        num_envs:         batch size (defaults to ``min(n_games, 256)``).
        max_steps_per_game: hard cap to prevent runaway games.
        verbose:          print progress every 100 games.

    Returns:
        :class:`TournamentResult` with win rates and confidence intervals.
    """
    if num_envs is None:
        num_envs = min(n_games, 256)
    num_envs = max(1, min(num_envs, n_games))

    # Seat plan: which seats each game assigns to A vs B.
    if seats_per_game is None:
        seats_per_game = [["a", "b", "a", "b"] for _ in range(n_games)]
    seats_per_game = list(seats_per_game)
    if len(seats_per_game) != n_games:
        raise ValueError("seats_per_game length must equal n_games")

    seat_a_total = [s.count("a") for s in seats_per_game]
    seat_b_total = [s.count("b") for s in seats_per_game]
    if any(a + b != 4 for a, b in zip(seat_a_total, seat_b_total)):
        raise ValueError("each game's seat plan must have exactly 4 entries")

    # Build the BatchedEnv.
    env = fc.BatchedEnv(num_envs=num_envs, seed=seed)
    env.reset()

    # Per-batch buffers.
    actions = np.zeros(num_envs, dtype=np.uint32)
    rewards = np.zeros(num_envs, dtype=np.float32)
    dones   = np.zeros(num_envs, dtype=np.uint8)
    masks   = np.zeros((num_envs, fc.MASK_WORDS), dtype=np.uint64)
    obs_buf = np.zeros(fc.OBS_SIZE, dtype=np.float32)

    # Per-slot bookkeeping (which game id is currently in each batch slot,
    # which seat plan that game uses, step count for that game).
    slot_to_game = list(range(num_envs))
    next_game_id = num_envs

    game_step_count = [0] * n_games          # steps used per game
    game_winner = [-1] * n_games             # 0..3 winner seat, -1 if not finished
    game_truncated = [False] * n_games

    games_finished = 0

    def is_game_a(game_idx: int, winner_seat: int) -> bool:
        return seats_per_game[game_idx][winner_seat] == "a"

    while games_finished < n_games:
        env.write_masks(masks)

        # For each slot, pick an action via the appropriate policy.
        for i in range(num_envs):
            game_idx = slot_to_game[i]
            if game_idx is None:
                actions[i] = 0
                continue

            # Skip slots whose game already finished (waiting to be repurposed).
            if game_winner[game_idx] != -1 or game_truncated[game_idx]:
                actions[i] = 0
                continue

            # Cap on game length.
            if game_step_count[game_idx] >= max_steps_per_game:
                game_truncated[game_idx] = True
                games_finished += 1
                actions[i] = 0
                continue

            seat = env.current_player(i)
            seats = seats_per_game[game_idx]
            chosen_agent = agent_a if seats[seat] == "a" else agent_b

            # Render obs from this seat's POV.
            env.write_obs_pov(i, seat, obs_buf)
            mask_view = masks[i]

            try:
                a = int(chosen_agent(obs_buf, mask_view, i, seat, env))
            except Exception as e:
                raise RuntimeError(
                    f"agent threw exception at game {game_idx}, slot {i}, seat {seat}: {e}"
                ) from e
            actions[i] = a

        env.step(actions, rewards, dones)

        # Process step results.
        for i in range(num_envs):
            game_idx = slot_to_game[i]
            if game_idx is None:
                continue
            game_step_count[game_idx] += 1

            if dones[i]:
                winner_seat = env.last_winner(i)
                if winner_seat < fc.NUM_PLAYERS:
                    game_winner[game_idx] = int(winner_seat)
                games_finished += 1
                if verbose and games_finished % 100 == 0:
                    print(f"  finished {games_finished}/{n_games} games")
                # Slot is auto-reset by BatchedEnv. Repurpose for next game.
                if next_game_id < n_games:
                    slot_to_game[i] = next_game_id
                    next_game_id += 1
                else:
                    slot_to_game[i] = None
            elif game_truncated[game_idx]:
                # Truncation already counted; clear slot.
                if next_game_id < n_games:
                    slot_to_game[i] = next_game_id
                    next_game_id += 1
                else:
                    slot_to_game[i] = None

    # Aggregate.
    wins_a = wins_b = ties = truncated = 0
    completed_lengths = []
    for i in range(n_games):
        if game_truncated[i]:
            truncated += 1
            continue
        winner_seat = game_winner[i]
        if winner_seat < 0:
            ties += 1
            continue
        completed_lengths.append(game_step_count[i])
        if is_game_a(i, winner_seat):
            wins_a += 1
        else:
            wins_b += 1

    n_completed = wins_a + wins_b + ties
    win_rate_a = wins_a / n_completed if n_completed else 0.0
    win_rate_b = wins_b / n_completed if n_completed else 0.0
    ci_a = _wilson_ci(wins_a, n_completed)
    ci_b = _wilson_ci(wins_b, n_completed)
    avg_len = (sum(completed_lengths) / len(completed_lengths)
               if completed_lengths else 0.0)

    seat_a_used = [i for i, s in enumerate(seats_per_game[0]) if s == "a"]
    seat_b_used = [i for i, s in enumerate(seats_per_game[0]) if s == "b"]

    return TournamentResult(
        n_games=n_games,
        n_completed=n_completed,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        truncated=truncated,
        win_rate_a=win_rate_a,
        win_rate_b=win_rate_b,
        ci95_a=ci_a,
        ci95_b=ci_b,
        avg_game_length=avg_len,
        seat_a=seat_a_used,
        seat_b=seat_b_used,
    )
