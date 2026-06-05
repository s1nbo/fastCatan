"""Single-process GPU-batched AlphaZero self-play trainer.

Replaces train.py's W CPU worker processes with ONE process driving G
concurrent games through BatchedMCTS (models/alphazero/batched_mcts.py): the
GPU does both the batched leaf evals and the gradient steps, the C++
BatchedEnv does all game/scratch stepping in OpenMP, and Python only
orchestrates. Measured ~9-10x per-decision vs the per-game MCTS at
sims=128, flat in G (python-floor-bound, so bigger nets are ~free).

Game loop (lockstep move-steps over G games):
  1. fast-forward: games whose to-move seat has exactly one (p2p-filtered)
     legal action step it directly — no search, no record (same as
     selfplay.play_one_game). Repeats until every live game is at a
     multi-legal decision.
  2. search: ONE BatchedMCTS.search over all G roots -> visit policies.
  3. record + act: store (obs from to-move POV, pi, mask, seat); sample the
     move (temp=1 for the first --temp-moves decisions of each game, then
     greedy); step all games.
  4. finished games: read the winner from the signature VPs, assign z to the
     game's records (sparse +-1 or vp_margin), push to the replay buffer,
     reload that slot with a fresh game (manual reset — the game batch runs
     step_raw, so terminals are never silently wiped).
  5. train: every --train-every move-steps, --train-steps gradient steps on
     the GPU once the buffer holds --min-buffer samples.

Run (smoke):
    python -m models.alphazero.batched_selfplay --total-games 64 \
        --num-games 32 --sims 48 --device cuda
"""
from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import fastcatan

from models.ckpt import write_stamp, verify_stamp
from models.alphazero.net import PolicyValueNet, masked_log_softmax
from models.alphazero.batched_mcts import BatchedMCTS, SNAP, SKIP, WIN_VP
from models.alphazero.batched_eval import eval_vs_random_raw

OBS_SIZE = fastcatan.OBS_SIZE
NUM_ACTIONS = fastcatan.NUM_ACTIONS
MASK_WORDS = fastcatan.MASK_WORDS
NUM_PLAYERS = fastcatan.NUM_PLAYERS
SIG_INTS = fastcatan.SIG_INTS

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


class GameSlot:
    """Per-slot record state for one live game."""
    __slots__ = ("records", "decisions")

    def __init__(self):
        self.records: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self.decisions = 0


class BatchedSelfplay:
    def __init__(self, net, args):
        self.args = args
        self.G = args.num_games
        self.net = net
        self.device = args.device
        self.suppress = not args.allow_trades

        self.mcts = BatchedMCTS(
            net, self.G, device=args.device, sims=args.sims,
            c_puct=args.c_puct, seed=args.seed, suppress_p2p=self.suppress)

        self.game = fastcatan.BatchedEnv(self.G, args.seed ^ 0x6A3E5)
        self.game.reset()
        self.scratch = fastcatan.Env()          # fresh-game factory
        self.seed_seq = random.Random(args.seed ^ 0x5EED)

        G = self.G
        self.snaps = np.zeros((G, SNAP), dtype=np.uint8)
        self.acts = np.zeros(G, dtype=np.uint32)
        self.rew = np.zeros(G, dtype=np.float32)
        self.done = np.zeros(G, dtype=np.uint8)
        self.masks_u64 = np.zeros((G, MASK_WORDS), dtype=np.uint64)
        self.sigs = np.zeros((G, SIG_INTS), dtype=np.int32)
        self.povs = np.zeros(G, dtype=np.uint8)
        self.obs = np.zeros((G, OBS_SIZE), dtype=np.float32)

        self.slots = [GameSlot() for _ in range(G)]
        self.buffer: deque = deque(maxlen=args.buffer_size)
        self.games_done = 0
        self.decisions_done = 0
        self.winners_hist: deque = deque(maxlen=256)

    # -------- masks / legality --------

    def _legal_bool(self) -> np.ndarray:
        self.game.write_masks(self.masks_u64)
        return self.mcts._filter(self.mcts._unpack_masks(self.masks_u64))

    # -------- finished-game handling --------

    def _finish_game(self, g: int) -> None:
        sig = self.sigs[g]
        vps = sig[8:12]
        winner = -1
        for p in range(NUM_PLAYERS):
            if vps[p] >= WIN_VP:
                winner = p
                break
        slot = self.slots[g]
        for obs, pi, mask, seat in slot.records:
            if self.args.value_mode == "vp_margin":
                best_other = max(int(vps[q]) for q in range(NUM_PLAYERS)
                                 if q != seat)
                z = float(np.clip((int(vps[seat]) - best_other) / 10.0,
                                  -1.0, 1.0))
            else:
                z = 1.0 if seat == winner else -1.0
            self.buffer.append((obs, pi, mask, z))
        self.games_done += 1
        self.winners_hist.append(winner)
        slot.records = []
        slot.decisions = 0
        # reload the slot with a fresh game
        self.scratch.reset(self.seed_seq.getrandbits(64))
        self.snaps[g] = np.frombuffer(self.scratch.snapshot(), dtype=np.uint8)

    def _apply_dones(self) -> bool:
        """Harvest finished games; returns True if any slot was reloaded."""
        if not self.done.any():
            return False
        self.game.write_sigs(self.sigs)
        self.game.save_snapshots(self.snaps)
        for g in np.nonzero(self.done)[0]:
            self._finish_game(int(g))
        self.game.load_snapshots(self.snaps)
        return True

    # -------- the lockstep move-step --------

    def _fast_forward(self) -> None:
        """Step every game whose to-move seat is forced (single legal action)
        until all G games sit at a multi-legal decision point."""
        for _ in range(512):                     # forced runs are short
            legal = self._legal_bool()
            n_legal = legal.sum(axis=1)
            forced = n_legal == 1
            if not forced.any():
                return
            self.acts[:] = SKIP
            forced_idx = np.nonzero(forced)[0]
            self.acts[forced_idx] = legal[forced_idx].argmax(axis=1)
            self.game.step_raw(self.acts, self.rew, self.done)
            self._apply_dones()
        raise RuntimeError("fast-forward failed to reach decision points")

    def move_step(self, temp_moves: int) -> int:
        """One search+act cycle across all G games. Returns decisions made."""
        self._fast_forward()
        self.game.save_snapshots(self.snaps)

        pi, mask, tm = self.mcts.search(self.snaps, add_root_noise=True)

        # record from each to-move seat's POV
        np.copyto(self.povs, tm.astype(np.uint8))
        self.game.write_obs_pov_batch(self.povs, self.obs)
        temps = np.array(
            [1.0 if self.slots[g].decisions < temp_moves else 0.0
             for g in range(self.G)], dtype=np.float64)
        acts = self.mcts.choose(pi, mask, temps)
        for g in range(self.G):
            self.slots[g].records.append(
                (self.obs[g].copy(), pi[g].copy(), mask[g].copy(), int(tm[g])))
            self.slots[g].decisions += 1
        self.decisions_done += self.G

        self.game.step_raw(acts, self.rew, self.done)
        self._apply_dones()
        return self.G

    # -------- training --------

    def train_steps(self, opt, n_steps: int) -> dict:
        self.net.train()
        last = {"loss": float("nan"), "policy": float("nan"),
                "value": float("nan")}
        for _ in range(n_steps):
            batch = random.sample(self.buffer,
                                  min(self.args.batch_size, len(self.buffer)))
            obs = torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device)
            pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(self.device)
            mask = torch.from_numpy(np.stack([b[2] for b in batch])).to(self.device)
            z = torch.tensor([b[3] for b in batch], dtype=torch.float32,
                             device=self.device)
            logits, value = self.net(obs)
            logp = masked_log_softmax(logits, mask)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            value_loss = F.mse_loss(value, z)
            loss = policy_loss + self.args.value_coef * value_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()
            last = {"loss": float(loss), "policy": float(policy_loss),
                    "value": float(value_loss)}
        self.net.eval()
        return last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-games", type=int, default=20000)
    p.add_argument("--num-games", type=int, default=256,
                   help="G: concurrent games searched in lockstep.")
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temp-moves", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-every", type=int, default=2,
                   help="move-steps between training bursts.")
    p.add_argument("--train-steps", type=int, default=8,
                   help="gradient steps per burst.")
    p.add_argument("--min-buffer", type=int, default=5000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--value-coef", type=float, default=1.0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str,
                   default=str(CKPT_DIR / "alphazero_batched"))
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="games between checkpoints.")
    p.add_argument("--eval-every", type=int, default=1000,
                   help="games between raw-policy vs-random evals (0=off).")
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument("--allow-trades", action="store_true")
    p.add_argument("--value-mode", choices=["sparse", "vp_margin"],
                   default="sparse")
    p.add_argument("--init-from", type=str, default="",
                   help="Warm-start net weights from this checkpoint (.pt).")
    p.add_argument("--hidden", type=str, default="512,512,256",
                   help="comma-separated trunk widths (GPU batching makes "
                        "bigger nets ~free; see scaling roadmap step 3).")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    net = PolicyValueNet(hidden=hidden).to(args.device)
    if args.init_from:
        verify_stamp(args.init_from, strict=False)
        state = torch.load(args.init_from, map_location=args.device,
                           weights_only=False)
        net.load_state_dict(state["net_state"])
        print(f"[init] warm-started from {args.init_from}", flush=True)
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    sp = BatchedSelfplay(net, args)
    print(f"[cfg] G={args.num_games} sims={args.sims} device={args.device} "
          f"hidden={hidden} trades={'on' if args.allow_trades else 'off'} "
          f"value={args.value_mode} total={args.total_games}", flush=True)

    t0 = time.time()
    it = 0
    last_ckpt = 0
    last_eval = 0
    stats = {"loss": float("nan"), "policy": float("nan"),
             "value": float("nan")}
    while sp.games_done < args.total_games:
        it += 1
        sp.move_step(args.temp_moves)
        if (len(sp.buffer) >= args.min_buffer
                and it % args.train_every == 0):
            stats = sp.train_steps(opt, args.train_steps)

        if it % 20 == 0:
            el = time.time() - t0
            wl = list(sp.winners_hist)
            n_won = sum(1 for w in wl if w >= 0)
            print(f"[it {it:>5d}] games={sp.games_done} "
                  f"buf={len(sp.buffer)} "
                  f"dec/s={sp.decisions_done/el:7.1f} "
                  f"g/s={sp.games_done/el:6.3f} "
                  f"won={n_won}/{len(wl)} "
                  f"loss={stats['loss']:.3f} p={stats['policy']:.3f} "
                  f"v={stats['value']:.3f}", flush=True)

        if args.eval_every and sp.games_done - last_eval >= args.eval_every:
            last_eval = sp.games_done
            te = time.time()
            wr, nw = eval_vs_random_raw(
                net, args.device, games=args.eval_games,
                seed=args.seed + 777 + sp.games_done,
                suppress_p2p=sp.suppress)
            print(f"   [eval g={sp.games_done}] raw-policy vs-random "
                  f"{wr:.3f} (no-winner {nw}, {time.time()-te:.1f}s)",
                  flush=True)

        if sp.games_done - last_ckpt >= args.checkpoint_every:
            last_ckpt = sp.games_done
            ck = save_dir / f"az_g{sp.games_done}.pt"
            torch.save({"net_state": net.state_dict(), "args": vars(args),
                        "games": sp.games_done}, str(ck))
            write_stamp(ck)
            print(f"[ckpt] {ck}", flush=True)

    final = save_dir / "az_final.pt"
    torch.save({"net_state": net.state_dict(), "args": vars(args),
                "games": sp.games_done}, str(final))
    write_stamp(final)
    el = time.time() - t0
    print(f"[done] {sp.games_done} games, {sp.decisions_done} decisions in "
          f"{el:.0f}s ({sp.games_done/el:.3f} g/s) -> {final}", flush=True)


if __name__ == "__main__":
    main()
