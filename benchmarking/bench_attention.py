import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from flash_attention.flash_attention import FlashAttentionTriton


def _maybe_compile(fn):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn)
        except Exception:
            return fn
    return fn


def _saved_activations_mib(output):
    grad_fn = getattr(output, "grad_fn", None)
    if grad_fn is None or not hasattr(grad_fn, "saved_tensors"):
        return 0.0
    total_bytes = sum(t.numel() * t.element_size() for t in grad_fn.saved_tensors)
    return total_bytes / (1024**2)


def _time_forward(fn, q, k, v, nb_warmup, nb_passes):
    for _ in range(nb_warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(nb_passes):
        out = fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / nb_passes, torch.cuda.max_memory_allocated() / (1024**2), out


def _time_backward(fn, q, k, v, nb_warmup, nb_passes):
    for _ in range(nb_warmup):
        q_w = q.detach().clone().requires_grad_(True)
        k_w = k.detach().clone().requires_grad_(True)
        v_w = v.detach().clone().requires_grad_(True)
        fn(q_w, k_w, v_w).sum().backward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(nb_passes):
        q_b = q.detach().clone().requires_grad_(True)
        k_b = k.detach().clone().requires_grad_(True)
        v_b = v.detach().clone().requires_grad_(True)
        out = fn(q_b, k_b, v_b)
        grad = torch.randn_like(out)
        out.backward(grad)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / nb_passes, torch.cuda.max_memory_allocated() / (1024**2)


def _make_impls():
    return {
        "pytorch_sdpa": _maybe_compile(
            lambda q, k, v: F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=False
            ).squeeze(1)
        ),
        "flash_triton": _maybe_compile(
            lambda q, k, v: FlashAttentionTriton.apply(q, k, v, False)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl",
        choices=["pytorch_sdpa", "flash_triton"],
        default=None,
        help="Run only one implementation",
    )
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float32
    batch_size = 8
    nb_warmup = 10
    nb_forward_passes = 100
    nb_backward_passes = 100
    d_models = [64]
    context_lengths = [256, 1024, 4096, 8192, 16384]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")

    impls = _make_impls()
    if args.impl is not None:
        impls = {args.impl: impls[args.impl]}

    results = []
    output_path = Path("outputs/csv/attention_benchmark.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0)

    for d_model in d_models:
        for seq_len in context_lengths:
            q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

            for name, fn in impls.items():
                record = {
                    "implementation": name,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "forward_ms": None,
                    "forward_peak_MiB": None,
                    "backward_ms": None,
                    "backward_peak_MiB": None,
                    "saved_activations_MiB": None,
                    "status": "ok",
                    "gpu": gpu_name,
                }

                try:
                    forward_ms, forward_peak_mib, out = _time_forward(
                        fn, q, k, v, nb_warmup, nb_forward_passes
                    )
                    record["forward_ms"] = forward_ms
                    record["forward_peak_MiB"] = forward_peak_mib
                    record["saved_activations_MiB"] = _saved_activations_mib(out)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        record["status"] = "OOM"
                        torch.cuda.empty_cache()
                        results.append(record)
                        continue
                    raise

                try:
                    backward_ms, backward_peak_mib = _time_backward(
                        fn, q, k, v, nb_warmup, nb_backward_passes
                    )
                    record["backward_ms"] = backward_ms
                    record["backward_peak_MiB"] = backward_peak_mib
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        record["status"] = "OOM(backward)"
                        torch.cuda.empty_cache()
                    else:
                        raise

                results.append(record)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
