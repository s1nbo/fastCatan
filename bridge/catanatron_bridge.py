"""CatanatronBridge: wraps a fastcatan-trained policy as a catanatron Player.

Thesis eval path:
    - Train agent in fastcatan (fast C++ sim).
    - Plug agent's policy fn into this bridge.
    - Run inside catanatron's reference engine vs catanatron baselines
      (AlphaBetaPlayer, ValueFunctionPlayer, etc.).
    - Numbers comparable to Catanatron paper.

Policy signature:
    policy(obs, mask, rng) -> int
        obs:  np.ndarray[float32, (OBS_SIZE,)]   from encode_obs
        mask: list[int]                          sorted list of legal fast IDs
        rng:  random.Random                      seeded per bridge

Default policy is uniform-over-mask (smoke-test pipeline). Plug NN here once
trained.

Design note (deviation from plan):
    Plan called for a fastcatan.Env state-mirror inside the bridge to source
    both obs and mask. We don't mirror because fastcatan's dice/dev/steal RNG
    can't be sync'd with catanatron's without a C++ patch (force_dice etc.).
    Instead:
      - obs from encode_obs(game, color) directly off the catanatron state
      - mask from catanatron's playable_actions reverse-mapped to fast IDs
      - composed actions (MOVE_ROBBER) handled via 2-step sub-policy pattern
      - trade composition (OFFER_TRADE / CONFIRM_TRADE) skipped — run cat
        eval with --no-player-trading for now. Full trade support requires
        the mirror+force-dice C++ patch.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

from catanatron import Color
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Player

import fastcatan
from bridge import topology_map as T
from bridge.action_codec import encode_to_fast_ids
from bridge.obs_encoder import encode_obs


_a = fastcatan.action

# Sentinel fast-ID for "no victim" branch of MOVE_ROBBER. Out of fastcatan's
# 286-ID range; used only internally inside the 2-step sub-decision.
_STEAL_NONE_SENTINEL = _a.STEAL_BASE + 4


PolicyFn = Callable[[np.ndarray, "list[int]", random.Random], int]


def uniform_policy(obs: np.ndarray, mask: "list[int]", rng: random.Random) -> int:
    """Uniform random over the legal fast-ID mask. Default smoke-test policy."""
    return rng.choice(mask)


class CatanatronBridge(Player):
    """Wraps a fastcatan-style policy as a Catanatron Player.

    `policy` defaults to `uniform_policy` so the bridge plugs in immediately
    for smoke tests / pipeline checks. Swap in an NN-backed policy once the
    agent is trained.
    """

    def __init__(self, color: Color, policy: Optional[PolicyFn] = None,
                 seed: int = 0):
        super().__init__(color)
        self._rng = random.Random(seed)
        self._policy = policy if policy is not None else uniform_policy

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        # MOVE_ROBBER must be returned as a single cat Action carrying both
        # (coord, victim). fastcatan splits this across MOVE_ROBBER_BASE+hex
        # and STEAL_BASE+seat IDs, so we run a 2-step sub-policy below.
        move_robbers = [a for a in playable_actions
                        if a.action_type == ActionType.MOVE_ROBBER]
        if move_robbers:
            obs = encode_obs(game, self.color)
            return self._decide_move_robber(
                obs, move_robbers, game.state.color_to_index)

        # Regular path: build {fast_id: cat_action} mapping for non-trade
        # actions, mask becomes the keyset, policy picks one.
        rep_map: dict[int, Action] = {}
        skipped: list[Action] = []
        for a in playable_actions:
            at = a.action_type
            if at in (ActionType.OFFER_TRADE,
                      ActionType.CONFIRM_TRADE,
                      ActionType.ACCEPT_TRADE,
                      ActionType.REJECT_TRADE,
                      ActionType.CANCEL_TRADE):
                # Trade composition needs scratch state; skip in MVP.
                skipped.append(a)
                continue
            try:
                fids = encode_to_fast_ids(a)
            except (KeyError, ValueError):
                skipped.append(a)
                continue
            if not fids:
                skipped.append(a)
                continue
            # Length-1 sequences (the common case). For longer sequences
            # (would be OFFER_TRADE), we'd skip earlier.
            rep_map.setdefault(fids[0], a)

        if not rep_map:
            # Nothing maps cleanly — fall back to uniform over the raw
            # playable list. Keeps games unblocked even in degenerate cases.
            return self._rng.choice(playable_actions)

        obs = encode_obs(game, self.color)
        mask = sorted(rep_map.keys())
        chosen = self._policy(obs, mask, self._rng)
        if chosen not in rep_map:
            chosen = self._rng.choice(mask)
        return rep_map[chosen]

    def _decide_move_robber(self, obs: np.ndarray, move_robbers, color_to_seat):
        # Step 1: pick hex.
        hex_to_actions: dict[tuple, list[Action]] = {}
        for a in move_robbers:
            coord = a.value[0]
            hex_to_actions.setdefault(coord, []).append(a)

        hex_id_to_coord: dict[int, tuple] = {}
        for coord in hex_to_actions:
            fid = _a.MOVE_ROBBER_BASE + T.COORD_TO_FAST_HEX[coord]
            hex_id_to_coord[fid] = coord
        hex_mask = sorted(hex_id_to_coord.keys())

        chosen_hex_id = self._policy(obs, hex_mask, self._rng)
        if chosen_hex_id not in hex_id_to_coord:
            chosen_hex_id = self._rng.choice(hex_mask)
        chosen_coord = hex_id_to_coord[chosen_hex_id]

        candidates = hex_to_actions[chosen_coord]
        if len(candidates) == 1:
            return candidates[0]

        # Step 2: pick victim.
        victim_to_action: dict[Optional[Color], Action] = {}
        for a in candidates:
            victim_to_action[a.value[1]] = a

        steal_id_to_victim: dict[int, Optional[Color]] = {}
        for victim in victim_to_action:
            fid = (_STEAL_NONE_SENTINEL if victim is None
                   else _a.STEAL_BASE + color_to_seat[victim])
            steal_id_to_victim[fid] = victim
        steal_mask = sorted(steal_id_to_victim.keys())

        chosen_steal_id = self._policy(obs, steal_mask, self._rng)
        if chosen_steal_id not in steal_id_to_victim:
            chosen_steal_id = self._rng.choice(steal_mask)
        return victim_to_action[steal_id_to_victim[chosen_steal_id]]
