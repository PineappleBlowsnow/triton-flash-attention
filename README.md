# Implementing Tiled Attention in Triton

**MVA course — LLMs for Code and Proof**

A compact MVA coursework implementation of **tiled attention with online softmax**. The forward pass is implemented in Triton; the custom backward is implemented in PyTorch and reconstructs the full attention matrices.

**Author:** Ying Jin / [PineappleBlowsnow](https://github.com/PineappleBlowsnow). Developed within the [GPU / LLM Flash-Attention course scaffold](https://github.com/dataflowr/gpu_llm_flash-attention), with the supplied correctness tests credited below.

## What I implemented

- A tiled PyTorch reference forward that updates the running row maximum, softmax normalizer and output without retaining a full score matrix.
- A Triton forward kernel with query/key tiling, online softmax, causal masking, boundary masks and float32 accumulators.
- Custom autograd integration: saving Q, K, V, output and log-sum-exp, then recomputing attention scores for the gradients.
- A CUDA-event benchmark comparing the implementation with PyTorch scaled dot-product attention (SDPA).

The implementation files and benchmark were added in my classroom commits `96d7d5f` and `01c98c0`. Course exercises, test design and the underlying attention algorithm are not original research contributions. This is a learning project, not a production attention library or a claim of CUDA C++ development.

## How it works

For each query tile, the kernel visits key/value tiles and maintains a running maximum `m`, normalizer `l` and normalized output `O`. When a new maximum is seen, the previous state is rescaled before incorporating the new tile. This avoids storing the complete `Q @ K.T` matrix in the forward pass.

The backward uses the saved log-sum-exp to reconstruct probabilities, then computes `dV`, `dQ` and `dK` with PyTorch matrix operations. **It materializes full score/probability matrices and therefore has quadratic attention-memory cost.** It is not the tiled, memory-efficient FlashAttention backward.

Read [the implementation](flash_attention/flash_attention.py), especially `_flash_attention_forward_pytorch`, `flash_fwd_kernel` and `attention_backward_impl`.

## Run the selected correctness tests

Use a Linux environment with a CUDA-capable GPU, a compatible PyTorch CUDA build and Triton. Exact versions of the historical environment were not saved as a lockfile; the current dependency list is an installation starting point, not a guarantee across every combination.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the PyTorch build matching your CUDA environment first.
python -m pip install -e ".[test,benchmark]"
python -m pytest -q
```

The selected, course-provided tests compare outputs and gradients with ordinary attention, including causal cases. They also check the saved log-sum-exp tensor. They are credited to the upstream course, not presented as tests I authored. GPU tests skip when CUDA is unavailable, and some unsupported-architecture compilation failures are explicitly skipped by the original fixture.

The module imports Triton at import time, so the supplied CPU-reference tests still require an environment where Triton can be imported. This release does not promise native Windows CPU-only execution.

Expected input format is `(batch, sequence, head_dim)`; flatten heads into the batch dimension yourself if needed. Use matching Q/K/V head dimensions and the tested dimensions 64 or 128. Arbitrary head dimensions, grouped-query attention, dropout and production mixed-precision coverage have not been established.

## Historical evidence

| Evidence | What it supports | Important limit |
|---|---|---|
| [Historical test summary](results/historical/test_summary.json) | The original XML records **43 passed**, no failures/errors/skips | Includes other course exercises omitted from this package; not a fresh run or the test count of this subset |
| [Benchmark CSV](results/historical/attention_benchmark.csv) | Saved forward/backward timing and memory observations on an NVIDIA RTX PRO 6000 Blackwell Server Edition | One historical setup; backend and software versions were not fully recorded |

The historical benchmark uses float32, batch size 8, head dimension 64 and noncausal attention, with sequence lengths from 256 to 16,384. For illustration, at length 4,096 the saved forward times are 0.326 ms for this Triton implementation and 0.846 ms for the selected PyTorch SDPA execution. The recorded backward-region peak allocations are approximately 1,160 MiB and 184 MiB respectively, consistent with the much less memory-efficient backward here. These are observations from that run, not general speedup or memory guarantees.

### Benchmark caveats

- SDPA's backend is not pinned. A comparison with this execution is not a comparison against every PyTorch/FlashAttention backend.
- The column called `backward_ms` includes cloning inputs, a forward pass and random-gradient creation in the timed loop. It is not isolated backward-kernel latency.
- Memory columns use absolute CUDA peak allocation, not incremental operator workspace; allocator state and live inputs affect the measurement.
- The saved-activation column is zero because the forward inputs do not require gradients in that benchmark path. Zero does not mean the training implementation needs no saved activations.
- `torch.compile` is attempted; compiler/backend/version differences can change the result. No hardware-wide performance conclusion is claimed.

To reproduce the historical protocol on suitable hardware:

```bash
python -m benchmarking.bench_attention
# Or run one path:
python -m benchmarking.bench_attention --impl flash_triton
```

New results are written to the ignored `outputs/csv/` directory. The historical CSV is retained unchanged under `results/historical/`.

## Publication validation

On 2026-09-26, the portfolio copy was checked for Python syntax, TOML/JSON validity, source hashes, test-result structure and benchmark dimensions. The packaging environment has no PyTorch, Triton or pytest installation and no detected CUDA tool. **No attention computation, GPU test or benchmark was rerun during publication.** See [VALIDATION.md](VALIDATION.md).

## Scope and attribution

- Source: [classroom repository](https://github.com/llm-2026/flash-attention-PineappleBlowsnow), based on the [course repository](https://github.com/dataflowr/gpu_llm_flash-attention).
- Algorithm reference: [FlashAttention-2](https://arxiv.org/abs/2307.08691), linked by the course.
- [Apache-2.0 license](LICENSE) and [NOTICE](NOTICE) are retained. File-level origin and hashes are recorded in [PROVENANCE.md](PROVENANCE.md) and [provenance.json](provenance.json).
- The fused softmax-matmul solution, full lecture notebooks, grading/submission machinery, uncommitted explanatory drafts, machine-specific XML metadata and hidden Git history are deliberately outside this package.

Useful next experiments are a genuinely tiled backward, non-aligned sequence-length tests, fixed SDPA backends and separately measured forward/backward latency. These are future work, not completed results.


[Back to the MVA project portfolio](https://github.com/PineappleBlowsnow/mva-portfolio)
