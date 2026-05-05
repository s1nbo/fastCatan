"""Self-play wrappers — turn a frozen RL policy into a tournament Policy.

Used to evaluate a trained MaskablePPO agent against itself, against
older checkpoints (PSRO-style league play), or as the opponent during a
fresh training run for the next iteration.

Two helpers are provided:

  - :func:`policy_from_sb3` — wrap a stable-baselines3 / sb3-contrib
    model so it satisfies the :data:`fastcatan.tournament.Policy`
    signature.

  - :class:`FrozenSelfPlayOpponent` — a callable suitable for
    ``fastcatan.GymEnv(opponent_fn=...)``. Wraps an SB3 model so the
    agent's training opponent is a snapshot of itself.

Example::

    from sb3_contrib import MaskablePPO
    import fastcatan as fc
    from fastcatan.selfplay import policy_from_sb3, FrozenSelfPlayOpponent

    # 1. Train against a frozen self-play opponent.
    model = MaskablePPO.load("checkpoints/iter0.zip")
    opp = FrozenSelfPlayOpponent(model, deterministic=False)
    env = fc.GymEnv(seat=0, opponent_fn=opp)
    # ... continue training new model against opp ...

    # 2. Evaluate via tournament harness.
    policy_a = policy_from_sb3(model_a, deterministic=True)
    policy_b = policy_from_sb3(model_b, deterministic=True)
    result = fc.play(agent_a=policy_a, agent_b=policy_b, n_games=200, seed=42)
"""
from __future__ import annotations
from typing import Optional

import numpy as np

import fastcatan as fc


def policy_from_sb3(model, deterministic: bool = True):
    """Wrap an SB3 / sb3-contrib model in the tournament Policy signature.

    Args:
        model:           ``MaskablePPO`` instance (or any model with
                         ``predict(obs, action_masks=..., deterministic=...)``)
        deterministic:   if True, ``predict`` returns the argmax; else samples.

    Returns:
        callable matching :data:`fastcatan.tournament.Policy`.
    """
    def policy(obs: np.ndarray, mask_packed: np.ndarray,
               env_idx: int, seat: int, env=None) -> int:
        # SB3 expects bool[NUM_ACTIONS]; unpack.
        bool_mask = fc.unpack_mask(mask_packed)
        action, _ = model.predict(obs, action_masks=bool_mask,
                                    deterministic=deterministic)
        # SB3 may return a 0-d ndarray, a 1-d ndarray, or a scalar.
        return int(np.asarray(action).flatten()[0])
    return policy


class FrozenSelfPlayOpponent:
    """Frozen-snapshot policy callable for ``GymEnv(opponent_fn=...)``.

    The opponent_fn signature for ``GymEnv`` is ``(obs, mask_packed) -> action``.
    This class adapts an SB3 model to that signature, unpacking the action
    mask on the fly.

    Args:
        model:           SB3 model with mask-aware ``predict``.
        deterministic:   if True (default), use greedy argmax.
    """

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = bool(deterministic)

    def __call__(self, obs: np.ndarray, mask_packed: np.ndarray) -> int:
        bool_mask = fc.unpack_mask(mask_packed)
        action, _ = self.model.predict(obs, action_masks=bool_mask,
                                          deterministic=self.deterministic)
        return int(np.asarray(action).flatten()[0])
