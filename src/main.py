from functools import lru_cache
from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.engine
import numpy as np
import onnxruntime as ort
import random
from .model import board_to_matrix
import pickle


TEMPERATURE = 1
move_to_index = pickle.load(open("./src/move_to_int.pkl", "rb"))


@lru_cache(maxsize=None)
def load_model(model_name):
    """INT8 ONNX Runtime session built by quantize.py; cached so each model is loaded once per process."""
    path = "./models/int8/" + model_name.replace(".pth", ".onnx")
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

        if 'pv' in model_name:

            # 2. Run your new search function to get the *single* best move
            best_move = shallow_search(board, model)

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



def shallow_search(board, model):
    """
    1-ply search using the value head.
    This is a "minimax" search at depth 1, scoring every child position in one batch.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    children = []
    for move in legal:
        board.push(move)
        children.append(board_to_matrix(board))
        board.pop()

    # Value is always from White's perspective, so Black maximizes its negation.
    white_scores = model.run(None, {"x": np.stack(children).astype(np.float32)})[1].ravel()
    my_scores = white_scores if board.turn == chess.WHITE else -white_scores
    return legal[int(my_scores.argmax())]



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
