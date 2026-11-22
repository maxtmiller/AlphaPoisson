"""Per-move latency of chess_manager.get_model_move, end to end.

Usage: python latency.py <repo_root> [model] [positions] [rounds]
Point <repo_root> at this checkout or at a worktree of an older commit to compare.
"""
import os
import random
import statistics
import sys
import time

import chess
import chess.pgn


def positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(16, 40)):
            board.push(rng.choice(list(board.legal_moves)))
            if board.is_game_over():
                break
        if not board.is_game_over():
            out.append(str(chess.pgn.Game.from_board(board)))
    return out


def main():
    repo = os.path.abspath(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "chess_resnet_op_elite_15.pth"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    # The old code loads models and pickles with paths relative to the repo root.
    os.chdir(repo)
    sys.path.insert(0, repo)
    from src.utils import chess_manager
    from src import main as _  # registers the entrypoint

    pgns = positions(n)
    chess_manager.set_context(pgns[0], 60000, model)
    chess_manager.get_model_move()

    medians = []
    for r in range(rounds):
        times = []
        for pgn in pgns:
            chess_manager.set_context(pgn, 60000, model)
            start = time.perf_counter()
            chess_manager.get_model_move()
            times.append((time.perf_counter() - start) * 1000)
        medians.append(statistics.median(times))
        print(f"round {r + 1}: median {medians[-1]:.2f} ms, p90 {statistics.quantiles(times, n=10)[-1]:.2f} ms")
    print(f"{model} @ {repo}: median of medians {statistics.median(medians):.2f} ms, load avg {os.getloadavg()[0]:.1f}")


if __name__ == "__main__":
    main()
