"""GPU-batched lockstep MCTS over the C++ fastcatan BatchedEnv.

mcts.MCTS runs one simulation at a time, so every leaf costs one tiny net
forward (batch 4 — the seat POVs). sims x decisions tiny forwards dominate the
wall clock, the GPU idles, and the net stays starved of games — the recurring
throughput wall (scaling-roadmap step 2 in the alphazero-v1 notes).

BatchedMCTS searches G independent root states IN LOCKSTEP:

  - All G trees advance one simulation together, one ply-round at a time.
  - Per round: PUCT selection for every still-descending game is one
    vectorized numpy pass; the selected actions advance through ONE
    ``BatchedEnv.step_raw`` (OpenMP, GIL released, no auto-reset); chance
    signatures come back from ONE ``write_sigs`` pass.
  - A game parks with ``SKIP_ACTION`` when its walk hits a new leaf or a
    terminal; the rest keep descending.
  - When every game is parked, ALL pending leaves are expanded with ONE net
    forward of shape (4·L, OBS_SIZE) — the GPU-batched leaf eval. Backups are
    max^n exactly as in mcts.py.

Tree semantics match mcts.MCTS: signature-keyed chance children, reseed before
every step, each round reloads the frontier node's CACHED snapshot (an
existing child replays its first-sample hidden state, same as the single-game
tree), p2p-trade suppression with the never-strand fallback, Dirichlet root
noise, visit-count policy targets. Only the schedule differs.
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
SNAP = fastcatan.SNAPSHOT_BYTES
SIG_INTS = fastcatan.SIG_INTS
SKIP = fastcatan.SKIP_ACTION
WIN_VP = 10

_SHIFTS = np.arange(64, dtype=np.uint64)


class Node:
    __slots__ = ("snap", "to_move", "mask", "P", "N", "W", "total_N", "children")

    def __init__(self, snap: np.ndarray, to_move: int, mask: np.ndarray):
        self.snap = snap                    # (SNAPSHOT_BYTES,) uint8, owned
        self.to_move = to_move
        self.mask = mask                    # (NUM_ACTIONS,) bool, owned
        self.P = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.W = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.total_N = 0
        # children[action] = {signature-bytes: Node}
        self.children: dict[int, dict[bytes, "Node"]] = {}


class BatchedMCTS:
    """Lockstep max^n MCTS for G concurrent games sharing one scratch
    BatchedEnv and one batched net."""

    def __init__(
        self,
        net: torch.nn.Module,
        num_games: int,
        device: str = "cuda",
        sims: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        seed: int = 0,
        suppress_p2p: bool = True,
    ):
        self.net = net
        self.G = num_games
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_frac = dirichlet_frac
        self._np_rng = np.random.default_rng(seed ^ 0xBA7C4ED)
        self._p2p = p2p_trade_mask() if suppress_p2p else None

        self.env = fastcatan.BatchedEnv(num_games, seed ^ 0x5C247C8)
        self.env.reset()
        G = num_games
        self._load_buf = np.zeros((G, SNAP), dtype=np.uint8)
        self._save_buf = np.zeros((G, SNAP), dtype=np.uint8)
        self._acts = np.zeros(G, dtype=np.uint32)
        self._rew = np.zeros(G, dtype=np.float32)
        self._done = np.zeros(G, dtype=np.uint8)
        self._masks_u64 = np.zeros((G, MASK_WORDS), dtype=np.uint64)
        self._sigs = np.zeros((G, SIG_INTS), dtype=np.int32)
        self._obs4 = np.zeros((G, 4, OBS_SIZE), dtype=np.float32)

    # -------- vectorized helpers --------

    def _unpack_masks(self, words: np.ndarray) -> np.ndarray:
        """(G, MASK_WORDS) uint64 -> (G, NUM_ACTIONS) bool, fully vectorized."""
        bits = (words[:, :, None] >> _SHIFTS[None, None, :]) & np.uint64(1)
        return bits.reshape(words.shape[0], -1)[:, :NUM_ACTIONS].astype(bool)

    def _filter(self, m: np.ndarray) -> np.ndarray:
        """AND-NOT p2p trades per row; never strand a row with no action."""
        if self._p2p is None:
            return m
        f = m & ~self._p2p[None, :]
        empty = ~f.any(axis=1)
        if empty.any():
            f[empty] = m[empty]
        return f

    @torch.no_grad()
    def _net_eval(self, obs4_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(L, 4, OBS) -> (logits (L, 4, A), values (L, 4)) as numpy."""
        t = torch.from_numpy(obs4_rows.reshape(-1, OBS_SIZE)).to(self.device)
        logits, values = self.net(t)
        L = obs4_rows.shape[0]
        return (logits.float().view(L, 4, -1).cpu().numpy(),
                values.float().view(L, 4).cpu().numpy())

    @staticmethod
    def _prior(row: np.ndarray, mask: np.ndarray) -> np.ndarray:
        row = np.where(mask, row, -np.inf)
        row = row - row.max()
        p = np.exp(row)
        p[~mask] = 0.0
        s = p.sum()
        return (p / s).astype(np.float32) if s > 0 else (
            mask.astype(np.float32) / mask.sum())

    def _noise(self, node: Node) -> None:
        legal = np.nonzero(node.mask)[0]
        if legal.size <= 1:
            return
        noise = self._np_rng.dirichlet([self.dir_alpha] * legal.size)
        f = self.dir_frac
        node.P[legal] = (1.0 - f) * node.P[legal] + f * noise.astype(np.float32)

    @staticmethod
    def _terminal_vec(sig_row: np.ndarray) -> np.ndarray:
        v = np.full(NUM_PLAYERS, -1.0, dtype=np.float32)
        vps = sig_row[8:12]
        for p in range(NUM_PLAYERS):
            if vps[p] >= WIN_VP:
                v[p] = 1.0
                break
        return v

    @staticmethod
    def _backup(path: list[tuple[Node, int]], vec: np.ndarray) -> None:
        for node, a in path:
            node.N[a] += 1
            node.W[a] += vec[node.to_move]
            node.total_N += 1

    # -------- one lockstep simulation across all active games --------

    def _one_sim(self, roots: list[Node | None], act_idx: np.ndarray) -> None:
        cur: dict[int, Node] = {int(g): roots[g] for g in act_idx}
        paths: dict[int, list[tuple[Node, int]]] = {int(g): [] for g in act_idx}
        pend: dict[int, tuple[Node, int, bytes, int]] = {}   # g -> (parent, a, sig, tm)
        term: dict[int, np.ndarray] = {}                      # g -> value vec
        alive = [int(g) for g in act_idx]

        rounds = 0
        while alive:
            rounds += 1
            if rounds > self.sims + 16:   # tree depth grows <= 1 per sim
                raise RuntimeError("batched MCTS descent failed to park")

            # frontier snapshots in, parked rows keep their leaf/terminal state
            for g in alive:
                self._load_buf[g] = cur[g].snap
            self.env.load_snapshots(self._load_buf)
            self.env.reseed(self._np_rng.integers(
                0, np.iinfo(np.int64).max, self.G, dtype=np.uint64))

            # vectorized PUCT over the alive frontier
            Pm = np.stack([cur[g].P for g in alive])
            Nm = np.stack([cur[g].N for g in alive]).astype(np.float32)
            Wm = np.stack([cur[g].W for g in alive])
            Mm = np.stack([cur[g].mask for g in alive])
            tot = np.array([cur[g].total_N for g in alive], dtype=np.float32)
            q = np.where(Nm > 0, Wm / np.maximum(Nm, 1.0), 0.0)
            u = self.c_puct * Pm * (np.sqrt(tot + 1.0)[:, None] / (1.0 + Nm))
            score = np.where(Mm, q + u, -np.inf)
            sel = score.argmax(axis=1)

            self._acts[:] = SKIP
            for k, g in enumerate(alive):
                self._acts[g] = sel[k]
            self.env.step_raw(self._acts, self._rew, self._done)
            self.env.write_sigs(self._sigs)
            self.env.save_snapshots(self._save_buf)

            nxt: list[int] = []
            for k, g in enumerate(alive):
                a = int(sel[k])
                node = cur[g]
                paths[g].append((node, a))
                if self._done[g]:
                    term[g] = self._terminal_vec(self._sigs[g])
                    self._load_buf[g] = self._save_buf[g]      # park on terminal
                    continue
                sig = self._sigs[g].tobytes()
                child = node.children.setdefault(a, {}).get(sig)
                if child is None:
                    pend[g] = (node, a, sig, int(self._sigs[g, 0]))
                    self._load_buf[g] = self._save_buf[g]      # park on leaf
                else:
                    cur[g] = child
                    nxt.append(g)
            alive = nxt

        # ---- batched leaf expansion: ONE forward for every pending leaf ----
        if pend:
            leaf_gs = np.fromiter(pend.keys(), dtype=np.int64)
            # parked slots still hold each leaf's state (SKIP + reload kept
            # them byte-stable), so one masks+obs pass reads them all.
            self.env.write_masks(self._masks_u64)
            self.env.write_obs_all4(self._obs4)
            leaf_masks = self._filter(self._unpack_masks(
                self._masks_u64[leaf_gs]))
            logits, values = self._net_eval(self._obs4[leaf_gs])
            for j, g in enumerate(leaf_gs):
                g = int(g)
                parent, a, sig, tm = pend[g]
                node = Node(self._load_buf[g].copy(), tm, leaf_masks[j].copy())
                node.P = self._prior(logits[j, tm], node.mask)
                parent.children[a][sig] = node
                self._backup(paths[g], values[j])
        for g, vec in term.items():
            self._backup(paths[g], vec)

    # -------- public API --------

    def search(
        self,
        roots_buf: np.ndarray,
        active: np.ndarray | None = None,
        add_root_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ``sims`` lockstep simulations from G root snapshots.

        roots_buf: (G, SNAPSHOT_BYTES) uint8 — row g is game g's root state.
        active:    optional bool (G,) — rows to search (others SKIP through).

        Returns (pi, mask, to_move): visit-count policy (G, NUM_ACTIONS)
        float32, legal mask (G, NUM_ACTIONS) bool, and the seat to move (G,)
        int8 (-1 on inactive rows). pi rows of inactive games are zero.
        """
        G = self.G
        assert roots_buf.shape == (G, SNAP)
        act_idx = (np.nonzero(active)[0] if active is not None
                   else np.arange(G))

        np.copyto(self._load_buf, roots_buf)        # every row a valid state
        self.env.load_snapshots(self._load_buf)
        self.env.write_masks(self._masks_u64)
        self.env.write_sigs(self._sigs)
        masks = self._filter(self._unpack_masks(self._masks_u64))

        roots: list[Node | None] = [None] * G
        for g in act_idx:
            g = int(g)
            roots[g] = Node(roots_buf[g].copy(), int(self._sigs[g, 0]),
                            masks[g].copy())

        if len(act_idx):
            self.env.write_obs_all4(self._obs4)
            logits, _values = self._net_eval(self._obs4[act_idx])
            for j, g in enumerate(act_idx):
                n = roots[int(g)]
                n.P = self._prior(logits[j, n.to_move], n.mask)
                if add_root_noise:
                    self._noise(n)

            for _ in range(self.sims):
                self._one_sim(roots, act_idx)

        pi = np.zeros((G, NUM_ACTIONS), dtype=np.float32)
        out_mask = np.zeros((G, NUM_ACTIONS), dtype=bool)
        to_move = np.full(G, -1, dtype=np.int8)
        for g in act_idx:
            g = int(g)
            n = roots[g]
            tot = n.N.sum()
            if tot > 0:
                pi[g] = n.N.astype(np.float32) / np.float32(tot)
            else:                                   # shouldn't happen
                pi[g] = n.mask.astype(np.float32) / n.mask.sum()
            out_mask[g] = n.mask
            to_move[g] = n.to_move
        return pi, out_mask, to_move

    def choose(
        self,
        pi: np.ndarray,
        mask: np.ndarray,
        temps: np.ndarray,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-game action selection from visit policies.

        temps: (G,) float — <=1e-6 means greedy argmax, else sample from
        counts ** (1/temp). Returns (G,) uint32 actions (SKIP on inactive).
        """
        G = self.G
        acts = np.full(G, SKIP, dtype=np.uint32)
        idx = np.nonzero(active)[0] if active is not None else np.arange(G)
        for g in idx:
            g = int(g)
            counts = pi[g].astype(np.float64)
            if temps[g] <= 1e-6:
                acts[g] = int(np.argmax(counts))
                continue
            heated = counts ** (1.0 / temps[g])
            s = heated.sum()
            probs = heated / s if s > 0 else (
                mask[g].astype(np.float64) / mask[g].sum())
            probs = probs / probs.sum()             # guard float drift
            acts[g] = int(self._np_rng.choice(NUM_ACTIONS, p=probs))
        return acts
