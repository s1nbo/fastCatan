"""M3 league: a bounded archive of past selves + PFSP-weighted matchmaking.

A drop-in alternative to `OpponentPool` (same `sample() -> {seat: opponent}` and
`random_opponent` surface, so `SelfPlayEnv` is unchanged). Instead of a sliding
window of the last N snapshots it keeps a **bounded archive of the `capacity`
best** snapshots and samples them by **Prioritized Fictitious Self-Play** (PFSP,
AlphaStar / Vinyals et al. 2019): an opponent's sampling weight rises the more
often it currently beats the learner, so training concentrates on the snapshots
the learner can't yet handle.

Per-member stat: the learner's smoothed win-rate
    p_i = (w_i + prior) / (n_i + 2*prior)            # Laplace prior -> 0.5 when unseen
fed live from training games — `SelfPlayEnv.step` credits every league opponent
that was at the table that game (a win iff seat 0 won). `--league-decay` fades
old counts each round so p_i tracks the *improving* learner.

PFSP weight over p:
    "hard"  w(p) = (1 - p)^beta     (default) — favor opponents you lose to
    "even"  w(p) = p * (1 - p)                — favor evenly-matched opponents

"Best `capacity`" (eviction when the archive overflows): the most-recent
`recent` snapshots are always kept (the newest are the strongest and a just-
frozen model has no stats yet); the remaining slots hold the hardest-for-the-
learner (lowest p_i), so the *easiest* non-recent member is evicted. This is a
bounded archive — unlike an unbounded league it can forget old easy-to-beat
strategies; that is the deliberate cost of the size cap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from models.selfplay.opponents import Opponent, PolicyOpponent, RandomOpponent


@dataclass
class _Member:
    opp: Opponent
    path: str
    w: float = 0.0  # learner wins vs this member (float so counts can decay)
    n: float = 0.0  # learner games vs this member


class League:
    """Bounded PFSP archive. Insertion order == recency (newest last)."""

    def __init__(
        self,
        seats: list[int],
        capacity: int = 32,
        recent: int = 8,
        p_random: float = 0.2,
        pfsp: str = "hard",
        beta: float = 2.0,
        prior: float = 1.0,
        seed: int = 0,
    ):
        if not 0 <= recent <= capacity:
            raise ValueError(f"need 0 <= recent({recent}) <= capacity({capacity})")
        if pfsp not in ("hard", "even"):
            raise ValueError(f"pfsp must be 'hard' or 'even', got {pfsp!r}")
        self.seats = list(seats)
        self.capacity = capacity
        self.recent = recent
        self.p_random = p_random
        self.pfsp = pfsp
        self.beta = beta
        self.prior = prior
        self._members: list[_Member] = []
        self._by_id: dict[int, _Member] = {}  # id(opp) -> member, for crediting
        self.random_opponent = RandomOpponent(seed)
        self._rng = random.Random(seed ^ 0x1EA6DE)

    # --- archive management ------------------------------------------------

    def add_candidate(self, opp: Opponent, path: str | Path) -> None:
        """Add a frozen snapshot, then evict the easiest non-recent member if the
        archive overflows."""
        m = _Member(opp=opp, path=str(path))
        self._members.append(m)
        self._by_id[id(opp)] = m
        self._evict_if_full()

    def _winrate(self, m: _Member) -> float:
        return (m.w + self.prior) / (m.n + 2 * self.prior)

    def _evict_if_full(self) -> None:
        while len(self._members) > self.capacity:
            # Protect the most-recent `recent`; among the rest evict the easiest
            # (highest learner win-rate == least useful as an opponent).
            evictable = self._members[: len(self._members) - self.recent]
            if not evictable:  # recent >= capacity: degenerate, spare only newest
                evictable = self._members[:-1]
            victim = max(evictable, key=self._winrate)
            self._members.remove(victim)
            self._by_id.pop(id(victim.opp), None)

    def decay_stats(self, factor: float) -> None:
        """Multiplicatively fade win/loss counts (e.g. 0.9) so PFSP weights track
        the improving learner. factor >= 1.0 is a no-op."""
        if factor >= 1.0:
            return
        for m in self._members:
            m.w *= factor
            m.n *= factor

    # --- matchmaking -------------------------------------------------------

    def _pfsp_weight(self, p: float) -> float:
        if self.pfsp == "hard":
            return (1.0 - p) ** self.beta
        return p * (1.0 - p)

    def _sample_member(self) -> _Member:
        weights = [self._pfsp_weight(self._winrate(m)) for m in self._members]
        total = sum(weights)
        if total <= 0.0:  # degenerate (shouldn't happen: prior keeps p in (0,1))
            return self._rng.choice(self._members)
        r = self._rng.random() * total
        upto = 0.0
        for m, wt in zip(self._members, weights):
            upto += wt
            if r <= upto:
                return m
        return self._members[-1]

    def sample(self) -> dict[int, Opponent]:
        """{seat: opponent} for one episode; each opponent seat independently
        gets the random opponent (prob `p_random`, anti-collapse) else a PFSP-
        weighted league member. No members yet -> all random (M2 regime)."""
        out: dict[int, Opponent] = {}
        for seat in self.seats:
            if not self._members or self._rng.random() < self.p_random:
                out[seat] = self.random_opponent
            else:
                out[seat] = self._sample_member().opp
        return out

    def record_result(self, opps, learner_won: bool) -> None:
        """Credit one finished game: each DISTINCT league opponent that was at the
        table gets +1 game (and +1 win iff the learner won). Non-members (the
        random opponent) and duplicate seats are ignored."""
        for oid in {id(o) for o in opps}:
            m = self._by_id.get(oid)
            if m is not None:
                m.n += 1.0
                if learner_won:
                    m.w += 1.0

    # --- OpponentPool-compatible surface -----------------------------------

    def __len__(self) -> int:
        return len(self._members)

    @property
    def snapshots(self) -> list[Opponent]:
        return [m.opp for m in self._members]

    @property
    def latest(self) -> Opponent:
        return self._members[-1].opp if self._members else self.random_opponent

    # --- persistence (for --resume) ----------------------------------------

    def state(self) -> dict:
        """Serializable archive snapshot: membership (in recency order) + counts."""
        return {
            "capacity": self.capacity,
            "recent": self.recent,
            "pfsp": self.pfsp,
            "beta": self.beta,
            "prior": self.prior,
            "members": [
                {"path": m.path, "name": m.opp.name, "w": m.w, "n": m.n}
                for m in self._members
            ],
        }

    def load_state(self, st: dict, device: str = "cpu") -> None:
        """Rebuild the archive from `state()`: reload each member from its path and
        restore its win/loss counts, preserving recency order."""
        self._members.clear()
        self._by_id.clear()
        for e in st.get("members", []):
            opp = PolicyOpponent.load(e["path"], name=e.get("name"), device=device)
            m = _Member(opp=opp, path=e["path"], w=float(e["w"]), n=float(e["n"]))
            self._members.append(m)
            self._by_id[id(opp)] = m
