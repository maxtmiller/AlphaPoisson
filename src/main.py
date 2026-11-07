from functools import lru_cache
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import chess.polyglot
import numpy as np
import time
import onnxruntime as ort
import random
from .board import board_to_matrix
import os
import pickle


TEMPERATURE = 1
# Outside the value head's [-1, 1] range, so a mate always beats a saturated network score.
MATE_SCORE = 2.0
MATE_PLY_PENALTY = 0.01
MAX_MOVE_SECONDS = 1.0
MAX_DEPTH = 10
QUIESCENCE_DEPTH = 6
MATERIAL_WEIGHT = 0.5
MATERIAL_SCALE = 5.0
POLICY_PRIOR = 0.06
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
# Policy output index of each move, per model file: each model family was trained with its own order.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
move_maps = pickle.load(open(os.path.join(ROOT, "src", "move_maps.pkl"), "rb"))


@lru_cache(maxsize=None)
def load_model(model_name):
    """INT8 ONNX Runtime session built by quantize.py; cached so each model is loaded once per process."""
    path = os.path.join(ROOT, "models", "int8", model_name.replace(".pth", ".onnx"))
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@chess_manager.entrypoint
def test_func(ctx: GameContext):        
    try:

        model_name = ctx.model_name
        print("Using model: ", model_name)
        
        board = ctx.board
        print("Cooking move...")
        print(board.move_stack)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (probably lost).")
        
        model = load_model(model_name)
        move_to_index = move_maps[model_name]

        if 'pv' in model_name:

            # 2. Search with the value and policy heads, leaving time for the rest of the game
            best_move = search(board, model, move_to_index, min(MAX_MOVE_SECONDS, ctx.timeLeft / 1000 / 30))

            # Failsafe
            if best_move is None:
                print("Search returned no move. Picking randomly.")
                best_move = random.choice(legal_moves)

            # 3. Create a "fake" probability map for logging
            ctx.logProbabilities({})

            print(f"Chosen move: {best_move.uci()}")
            return best_move
        
        elif 'op' in model_name:

            # 2. Convert board to tensor
            input_matrix = board_to_matrix(board)[np.newaxis].astype(np.float32)

            # 3. Run model
            logits = model.run(None, {"x": input_matrix})[0].flatten()
            exp = np.exp(logits - logits.max())
            all_move_probs = exp / exp.sum()

            # 4. Map legal moves to probabilities
            move_weights = []
            legal_moves_filtered = []
            for move in legal_moves:
                idx = move_to_index.get(move.uci(), None)
                if idx is not None:
                    prob = all_move_probs[idx]
                    move_weights.append(prob)
                    legal_moves_filtered.append(move)

            if not legal_moves_filtered:
                print("No legal moves found in move dictionary. Picking randomly.")
                best_move = random.choice(legal_moves)
                ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves})
                return best_move

            # 5. Normalize probabilities
            total_weight = sum(move_weights)
            if total_weight == 0:
                print("Model gave zero probability to all legal moves. Picking randomly.")
                best_move = random.choice(legal_moves_filtered)
                ctx.logProbabilities({m.uci(): 1.0 if m == best_move else 1e-6 for m in legal_moves_filtered})
                return best_move

            normalized_probs = {
                move.uci(): weight / total_weight
                for move, weight in zip(legal_moves_filtered, move_weights)
            }

            normalized_probs = {chess.Move.from_uci(k): float(v) for k, v in normalized_probs.items()}

            # 6. Log probabilities and choose move
            ctx.logProbabilities(normalized_probs)

            # Top K
            K = 3
            top_moves, top_probs = top_k_moves(normalized_probs, K)

            # Convert to dict for logging
            topk_dict = {m: p for m, p in zip(top_moves, top_probs)}

            ctx.logProbabilities(topk_dict)  # NOW logging matches the sampled distribution

            best_move = random.choices(top_moves, weights=top_probs, k=1)[0]

            return best_move
    
    except Exception as e:
        # --- THIS IS THE DEBUGGING PART ---
        print("!!!!!!!!!!!!!! ERROR IN TEST_FUNC !!!!!!!!!!!!!!")
        print(f"Exception Type: {type(e)}")
        print(f"Exception Args: {e.args}")
        
        # This prints the full traceback to your console
        import traceback
        traceback.print_exc()
        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Re-raise the exception so the server still fails
        raise e


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass



class SearchTimeout(Exception):
    pass


class AlphaBeta:
    """Negamax alpha-beta over the value head (side to move's perspective), ordered by the policy head."""

    def __init__(self, model, move_to_index, deadline):
        self.model = model
        self.move_to_index = move_to_index
        self.deadline = deadline
        self.can_stop = False
        self.cache = {}

    def evaluate(self, board):
        key = chess.polyglot.zobrist_hash(board)
        if key not in self.cache:
            policy, value = self.model.run(None, {"x": board_to_matrix(board)[np.newaxis].astype(np.float32)})
            # The value head barely tracks material (a whole knight moves it ~0.2), so blend in a material count.
            value = (1 - MATERIAL_WEIGHT) * float(value.ravel()[0]) + MATERIAL_WEIGHT * np.tanh(material(board) / MATERIAL_SCALE)
            self.cache[key] = (value, policy[0])
        return self.cache[key]

    def check_time(self):
        if self.can_stop and time.perf_counter() > self.deadline:
            raise SearchTimeout

    def ordered(self, board, policy):
        return sorted(board.legal_moves, reverse=True,
                      key=lambda m: policy[self.move_to_index[m.uci()]] if m.uci() in self.move_to_index else -np.inf)

    def negamax(self, board, depth, alpha, beta, ply):
        self.check_time()
        score = terminal_score(board, ply)
        if score is not None:
            return score
        if depth == 0:
            return self.quiescence(board, alpha, beta, ply, QUIESCENCE_DEPTH)
        best = -np.inf
        for move in self.ordered(board, self.evaluate(board)[1]):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def quiescence(self, board, alpha, beta, ply, depth):
        """Keep searching captures so a leaf is never scored in the middle of an exchange."""
        self.check_time()
        score = terminal_score(board, ply)
        if score is not None:
            return score
        best = self.evaluate(board)[0]
        if best >= beta or depth == 0:
            return best
        alpha = max(alpha, best)
        for move in sorted(board.generate_legal_captures(), key=lambda m: capture_order(board, m), reverse=True):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, depth - 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best


def search(board, model, move_to_index, seconds):
    """Iterative deepening: return the best move from the deepest search finished before the deadline."""
    engine = AlphaBeta(model, move_to_index, time.perf_counter() + seconds)
    root = board.copy()
    policy = engine.evaluate(root)[1]
    moves = engine.ordered(root, policy)
    # Root bonus from the policy's log-probabilities, so the value head can't overrule sound openings.
    logits = np.array([policy[move_to_index[m.uci()]] if m.uci() in move_to_index else -np.inf for m in moves])
    log_probs = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
    bonus = {m: POLICY_PRIOR * max(lp, -10.0) for m, lp in zip(moves, log_probs)}
    best_move, depth = moves[0], 0
    try:
        while depth < MAX_DEPTH:
            depth += 1
            # Depth 1 always finishes, so there is always a searched move to return.
            engine.can_stop = depth > 1
            alpha, iteration_best = -np.inf, moves[0]
            for move in moves:
                root.push(move)
                score = -engine.negamax(root, depth - 1, -np.inf, -(alpha - bonus[move]), 1)
                root.pop()
                # Scores past the value head's [-1, 1] range are mates, which the policy must not reorder.
                if abs(score) <= 1:
                    score += bonus[move]
                if score > alpha:
                    alpha, iteration_best = score, move
            best_move = iteration_best
            moves.remove(best_move)
            moves.insert(0, best_move)
    except SearchTimeout:
        depth -= 1
    print(f"Search depth {depth}, {len(engine.cache)} positions evaluated")
    return best_move


def material(board):
    """Material balance in pawns for the side to move."""
    return sum(PIECE_VALUES[p.piece_type] * (1 if p.color == board.turn else -1) for p in board.piece_map().values())


def capture_order(board, move):
    """Most valuable victim first, then least valuable attacker."""
    victim = board.piece_at(move.to_square)
    victim_type = victim.piece_type if victim else chess.PAWN  # en passant
    return victim_type * 10 - board.piece_at(move.from_square).piece_type


def terminal_score(board, ply):
    """Exact score for the side to move if the game is over, otherwise None. Nearer mates score higher."""
    if board.is_checkmate():
        return -MATE_SCORE + MATE_PLY_PENALTY * ply
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return 0.0
    return None



def top_k_moves(normalized_probs: dict, k: int):
    """
    Return the top-K moves and renormalized probabilities.
    normalized_probs: dict {Move: probability}
    """
    if k <= 0:
        raise ValueError("k must be >= 1")

    # Sort moves by probability (descending)
    sorted_moves = sorted(
        normalized_probs.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Take first k moves
    top_k = sorted_moves[:k]

    # Unpack moves + probabilities
    moves, probs = zip(*top_k)

    # Re-normalize probabilities to sum to 1
    total_prob = sum(probs)
    renormalized = [p / total_prob for p in probs]

    return list(moves), renormalized
