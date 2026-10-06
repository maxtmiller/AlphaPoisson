"""Latency of PyTorch FP32 vs ONNX Runtime INT8 (models/int8, from quantize.py). Run from the repo root."""
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

from quantize import build_model

MODELS = ["chess_resnet_op_bots_8.pth", "chess_resnet_op_elite_15.pth", "chess_resnet_op_elite_50.pth",
          "chess_resnet_pv_elite_1.pth", "chess_resnet_pv_elite_8.pth"]


def median_ms(fn, n=300):
    for _ in range(20):
        fn()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main():
    print(f"{'model':32} {'batch':>5} {'torch fp32':>11} {'ort int8':>9} {'speedup':>8}")
    for model_name in MODELS:
        fp32 = build_model(model_name)
        int8 = ort.InferenceSession("./models/int8/" + model_name.replace(".pth", ".onnx"), providers=["CPUExecutionProvider"])
        # Batch 32 approximates shallow search, which scores every legal reply at once.
        for batch in (1, 32):
            x = np.zeros((batch, 13, 8, 8), dtype=np.float32)
            xt = torch.from_numpy(x)
            with torch.no_grad():
                t32 = median_ms(lambda: fp32(xt))
            t8 = median_ms(lambda: int8.run(None, {"x": x}))
            print(f"{model_name:32} {batch:>5} {t32:>9.2f}ms {t8:>7.2f}ms {t32 / t8:>7.1f}x")


if __name__ == "__main__":
    main()
