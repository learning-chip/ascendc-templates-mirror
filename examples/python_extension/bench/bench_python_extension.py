import torch
import torch_npu
import torch_act
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

torch.ops.load_library("./output/python_extension/libact_torch.so")

def benchmark_and_plot():
    device = "npu:0"
    dtype_int = torch.int8
    dtype_fp16 = torch.float16
    dtype_bf16 = torch.bfloat16

    n_repeat = 100
    n_warmup = 5
    batch_size = 1
    scale_tensor = np.float32(1.0)

    sizes = [
        (512, 256, 1024),
        (1024, 2048, 8192),
        (64, 8192, 512),
        (128, 8192, 4096),
        (256, 8192, 2048),
        (512, 8192, 1024),
        (1024, 8192, 512),
        (2048, 8192, 256),
        (4096, 8192, 128),
        (8192, 8192, 64),
    ]

    low, high = -16, 16
    csv_path = "quant_matmul_benchmark_batch.csv"

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "M", "N", "K",
            "duration_batched_us",
            "duration_single_us",
            "duration_optimized_us",
            "duration_torch_mm_us"
        ])

        for M, K, N in sizes:
            print(f"\nBenchmarking M={M}, K={K}, N={N}")

            torch.manual_seed(0)

            # Input tensors
            a_int = torch.randint(low=low, high=high, size=(batch_size, M, K), dtype=dtype_int, device=device)
            b_int = torch.randint(low=low, high=high, size=(batch_size, K, N), dtype=dtype_int, device=device)
            bqmm_result = torch.zeros((batch_size, M, N), dtype=dtype_fp16, device=device)

            a_float = a_int[0].to(torch.float16)
            b_float = b_int[0].to(torch.float16)

            # Single-scale tensors
            per_token_scale = torch.ones(M, dtype=dtype_bf16, device=device)
            scale_bf16 = torch.ones(N, dtype=dtype_bf16, device=device)

            # Warm-up
            for _ in range(n_warmup):
                torch_act.batched_quant_matmul(a_int, b_int, bqmm_result, "float16", scale_tensor)
                _ = torch_act.quant_matmul(a_int[0], b_int[0], scale_bf16, per_token_scale, "bf16")
                _ = torch_act.optimized_quant_matmul(a_int[0], b_int[0], np.float32(1.0) , "float16")
                _ = torch.mm(a_float, b_float)

            # Timing
            def time_batched():
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(n_repeat):
                    torch_act.batched_quant_matmul(a_int, b_int, bqmm_result, "float16", scale_tensor)
                end.record()
                torch.npu.synchronize()
                return start.elapsed_time(end) / n_repeat * 1e3

            def time_single():
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(n_repeat):
                    _ = torch_act.quant_matmul(a_int[0], b_int[0], scale_bf16, per_token_scale, "bf16")
                end.record()
                torch.npu.synchronize()
                return start.elapsed_time(end) / n_repeat * 1e3
            
            def time_optimized_quant():
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(n_repeat):
                    _ = torch_act.optimized_quant_matmul(a_int[0], b_int[0], np.float32(1.0) , "float16")
                end.record()
                torch.npu.synchronize()
                return start.elapsed_time(end) / n_repeat * 1e3
            
            def time_torch_mm():
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(n_repeat):
                    _ = torch.mm(a_float, b_float)
                end.record()
                torch.npu.synchronize()
                return start.elapsed_time(end) / n_repeat * 1e3

            duration_batched = time_batched()
            duration_single = time_single()
            duration_optimized = time_optimized_quant()
            duration_torch_mm = time_torch_mm()

            print(f"batched_quant_matmul:\t{duration_batched:.2f} us")
            print(f"quant_matmul:\t\t{duration_single:.2f} us")
            print(f"optimized_quant_matmul:\t{duration_optimized:.2f} us")
            print(f"torch.mm:\t\t{duration_torch_mm:.2f} us")

            writer.writerow([
                M, N, K,
                f"{duration_batched:.6f}",
                f"{duration_single:.6f}",
                f"{duration_optimized:.6f}",
                f"{duration_torch_mm:.6f}"
            ])

if __name__ == "__main__":
    benchmark_and_plot()
