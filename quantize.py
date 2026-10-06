"""Post-training static INT8 quantization: models/<name>.pth -> models/int8/<name>.onnx (ONNX Runtime).

Calibration and evaluation positions come from seeded self-play with an FP32 policy model,
so they look like the positions the bots actually see. Run from the repo root: python quantize.py
"""
import os
import pickle
import random
import tempfile

import chess
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from src.model import ChessResNet, ChessResNetPa, board_to_matrix

MODELS_DIR = "./models"
OUT_DIR = "./models/int8"
CALIB_GAMES = 32
EVAL_GAMES = 8
# INT8 on the policy head's large Linear costs ~4 points of top-1 agreement for ~0.15 ms; keep it FP32.
FP32_NODES = ['/policy_fc/Gemm']

move_to_index = pickle.load(open("./src/move_to_int.pkl", "rb"))


def build_model(model_name):
    layers = int(model_name.split('.')[0].split('_')[-1])
    cls = ChessResNetPa if 'pv' in model_name else ChessResNet
    model = cls(num_res_blocks=layers, num_moves=1917)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, model_name), map_location='cpu'))
    return model.eval()


def self_play_positions(policy_model, games, seed):
    """Boards from games where each side samples a legal move from the policy (20% uniform random for variety)."""
    rng = random.Random(seed)
    boards = []
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < 120:
            boards.append(board.copy())
            legal = list(board.legal_moves)
            with torch.no_grad():
                x = torch.tensor(board_to_matrix(board), dtype=torch.float32).unsqueeze(0)
                probs = torch.softmax(policy_model(x), dim=1)[0]
            weights = [probs[move_to_index[m.uci()]].item() if m.uci() in move_to_index else 0.0 for m in legal]
            if rng.random() < 0.2 or sum(weights) == 0:
                move = rng.choice(legal)
            else:
                move = rng.choices(legal, weights=weights, k=1)[0]
            board.push(move)
    return boards


def to_batch(boards):
    return np.stack([board_to_matrix(b) for b in boards]).astype(np.float32)


def children(board):
    out = []
    for move in board.legal_moves:
        board.push(move)
        out.append(board.copy())
        board.pop()
    return out


class Reader(CalibrationDataReader):
    def __init__(self, x):
        self.batches = iter([{"x": x[i:i + 256]} for i in range(0, len(x), 256)])

    def get_next(self):
        return next(self.batches, None)


def export_int8(model_name, calib, out_path):
    model = build_model(model_name)
    model.fuse_model()
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = os.path.join(tmp, "fp32.onnx")
        torch.onnx.export(model, torch.zeros(1, 13, 8, 8), fp32_path, input_names=["x"],
                          dynamic_axes={"x": {0: "batch"}}, dynamo=False)
        quantize_static(fp32_path, out_path, Reader(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=FP32_NODES)


def legal_mask(boards):
    mask = np.full((len(boards), 1917), -np.inf, dtype=np.float32)
    for i, board in enumerate(boards):
        for move in board.legal_moves:
            if move.uci() in move_to_index:
                mask[i, move_to_index[move.uci()]] = 0
    return mask


def report(model_name, int8_path, eval_boards):
    fp32 = build_model(model_name)
    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    x = to_batch(eval_boards)
    with torch.no_grad():
        out32 = fp32(torch.from_numpy(x))
    out8 = sess.run(None, {"x": x})
    pv = 'pv' in model_name
    p32 = (out32[0] if pv else out32).numpy() + legal_mask(eval_boards)
    p8 = out8[0] + legal_mask(eval_boards)
    top1 = (p32.argmax(1) == p8.argmax(1)).mean()
    top3 = np.mean([len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) / 3 for a, b in zip(p32, p8)])
    msg = f"legal-move top-1 agreement {top1:.1%}, top-3 overlap {top3:.1%}"
    if pv:
        msg += f", value MAE {np.abs(out32[1].numpy() - out8[1]).mean():.4f}"
        same = 0
        for board in eval_boards:
            kids = to_batch(children(board))
            with torch.no_grad():
                v32 = fp32(torch.from_numpy(kids))[1].numpy().ravel()
            v8 = sess.run(None, {"x": kids})[1].ravel()
            sign = 1 if board.turn == chess.WHITE else -1
            same += (sign * v32).argmax() == (sign * v8).argmax()
        msg += f", shallow-search move agreement {same / len(eval_boards):.1%}"
    print(f"  {model_name}: {msg}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policy_model = build_model("chess_resnet_op_elite_50.pth")
    calib_boards = self_play_positions(policy_model, CALIB_GAMES, seed=0)
    eval_boards = self_play_positions(policy_model, EVAL_GAMES, seed=1)
    print(f"{len(calib_boards)} calibration positions, {len(eval_boards)} eval positions")

    calib = to_batch(calib_boards)
    # Shallow search scores every child position, so value models are calibrated on those too.
    calib_pv = to_batch(calib_boards + [c for b in random.Random(0).sample(calib_boards, 300) for c in children(b)])
    for model_name in sorted(f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')):
        out_path = os.path.join(OUT_DIR, model_name.replace('.pth', '.onnx'))
        export_int8(model_name, calib_pv if 'pv' in model_name else calib, out_path)
        report(model_name, out_path, eval_boards)


if __name__ == "__main__":
    main()
