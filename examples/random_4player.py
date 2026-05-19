import argparse
import random
import numpy as np

import fastcatan


def legal_actions(mask: np.ndarray) -> list[int]:
    """Return list of action IDs whose bit is set in the uint64 bitmask buffer."""
    out: list[int] = []
    for word_idx, word in enumerate(mask):
        w = int(word)
        base = word_idx * 64
        while w:
            bit = (w & -w).bit_length() - 1
            out.append(base + bit)
            w &= w - 1
    return out


def pick_random(rng: random.Random, mask: np.ndarray) -> int:
    legals = legal_actions(mask)
    if not legals:
        raise RuntimeError("no legal actions in mask")
    return rng.choice(legals)


def play_one(seed: int, max_steps: int, verbose: bool) -> tuple[int, list[int], int]:
    env = fastcatan.Env()
    env.reset(seed)

    mask = np.zeros(fastcatan.MASK_WORDS, dtype=np.uint64)
    rng = random.Random(seed)

    for step_idx in range(max_steps):
        env.action_mask(mask)
        action = pick_random(rng, mask)
        reward, done = env.step(action)

        if verbose and step_idx % 200 == 0:
            print(
                f"step={step_idx:5d} phase={env.phase} "
                f"cur={env.current_player} action={action} "
                f"vp={[env.player_vp(p) for p in range(4)]}"
            )

        if done:
            vps = [env.player_vp(p) for p in range(4)]
            winner = next((p for p, v in enumerate(vps) if v >= 10), -1)
            return winner, vps, step_idx + 1

    vps = [env.player_vp(p) for p in range(4)]
    return -1, vps, max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    win_counts = [0, 0, 0, 0]
    no_winner = 0
    total_steps = 0

    for g in range(args.games):
        winner, vps, steps = play_one(args.seed + g, args.max_steps, args.verbose)
        total_steps += steps
        if winner < 0:
            no_winner += 1
            print(f"game {g}: NO WINNER after {steps} steps  vps={vps}")
        else:
            win_counts[winner] += 1
            print(f"game {g}: winner={winner}  vps={vps}  steps={steps}")

    print()
    print(f"wins per seat: {win_counts}  no-winner: {no_winner}")
    print(f"avg steps/game: {total_steps / args.games:.1f}")


if __name__ == "__main__":
    main()
