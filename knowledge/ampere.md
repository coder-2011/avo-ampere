# Ampere Knowledge Base Seed

This is the initial static knowledge seed for the AVO agent. It is deliberately
small; the agent should consult external docs and local source reactively during
variation steps.

## Hardware

- Target GPU: NVIDIA RTX A6000, compute capability 8.6.
- Compile target: `sm_86`; use `-gencode=arch=compute_86,code=sm_86`.
- Ampere supports `cp.async` global-to-shared copies, `mma.sync` tensor-core
  instructions, warp-level shuffle/reduction patterns, and BF16 tensor cores.
- Ampere does not support Blackwell TMA, WGMMA, warp-group register allocation,
  or FA4's Blackwell-specific pipeline.
- Compute capability 8.6 has 100 KB shared memory per SM and 99 KB max shared
  memory per thread block, with explicit opt-in needed above 48 KB dynamic shared
  memory.
- NVIDIA's Ampere tuning guide says compute capability 8.6 has 48 resident warps
  per SM, a 64K 32-bit register file per SM, and a maximum of 16 resident thread
  blocks per SM. Treat register pressure and block count as first-class scoring
  risks, especially for head dimension 128.
- Ampere async global-to-shared copies can overlap memory movement with compute,
  avoid extra copy registers, and can bypass L1. This is the correct Ampere
  replacement direction for Blackwell-only TMA ideas.

## Baseline

- Seed and baseline should be FlashAttention-2.
- Correctness reference should be PyTorch `scaled_dot_product_attention`.
- Benchmark forward pass with BF16, head dimension 128, sequence lengths
  4096/8192/16384/32768, total tokens 32768, causal and non-causal.
- Baseline installation should pin FlashAttention compile targets for A6000 via
  `FLASH_ATTN_CUDA_ARCHS=80` (the upstream build script’s Ampere-family target)
  and cap build parallelism with `MAX_JOBS`.
- Score records can use `--trials N` to collect replicate CUDA-event timings.
  Per-case TFLOPS is computed from the median sample, and the JSON includes
  samples, min, mean, median, and coefficient of variation so noisy runs are
  visible instead of hidden.
- FlashAttention-2 v2.8.3 has a device-specific block-size heuristic for sm8x
  that treats sm86/sm89 separately from sm80. For head dimension 128, it chooses
  smaller N-blocks on sm86 in some cases: 64 for causal/no-dropout and 32 for
  non-causal/no-dropout. This is useful search-space evidence, not a commandment.
- Local candidates should currently start from
  `candidates/cuda_warp_rows_attention_seed.py` for tiny correctness smoke
  scoring. `cuda_tiled_attention_seed.py` is the one-CTA-per-row tiled reference.
  `cuda_naive_attention_seed.py` is the simpler one-thread-per-row attention
  reference. The older `cuda_identity_seed.py` is only an extension/build smoke
  because it delegates attention math to PyTorch SDPA before running a copy
  kernel.
- Use `evolve-once --attempts-dir ./attempts` during autonomous runs so rejected
  score attempts, command failures, and gate decisions remain available as recent
  prompt context without becoming committed lineage.

## Search Space

- `cp.async` pipeline depth and staging.
- A first inline `cp.async.cg.shared.global` staging attempt on the warp-row seed compiled to
  `LDGSTS.E.BYPASS.128` on sm86 and passed correctness, but regressed the tiny 64x64 BF16
  smoke versus synchronous shared staging. Do not re-enable a single-stage cp.async copy without
  adding actual overlap, double buffering, or profiler evidence.
- Shared-memory layouts and bank-conflict reduction.
- Register pressure and spill avoidance.
- Warp-level online softmax reductions.
- Instruction scheduling around `mma.sync` latency.
- Split-Q versus split-K work partitioning.
- The current warp-row seed uses four query rows per CTA, one warp per row,
  32-key score tiles, warp-shuffle max/sum reductions, FP32 row state, and
  online output rescaling. Its dot-product path uses a 4-wide packed load when
  head dimension is divisible by 4, with a scalar fallback for odd smoke shapes.
  It stages K/V tiles in shared memory only for head dimensions up to 64 and
  same-head CTAs; head dimension 128 and boundary CTAs use the global packed path.
  This is still far from FA2: it does not use `mma.sync` or `cp.async`.
- Next CUDA-kernel steps should keep correctness shapes small until row max,
  denominator, output accumulation, and causal masking are demonstrably correct
  for BF16 and FP32 before adding tensor-core or async-copy complexity.

## Sources

- NVIDIA Ampere tuning guide:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/
- FlashAttention-2 interface and sm8x block-size heuristic:
  https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py
- NVIDIA vectorized memory access guidance:
  https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- PyTorch benchmark utilities guidance on warmups, replicates, and median
  statistics:
  https://docs.pytorch.org/docs/stable/benchmark_utils.html

## Gate

A candidate enters lineage only when it passes correctness and matches or
improves the current best aggregate score. The score payload must also contain
at least one scored case and a finite positive geomean, so malformed empty score
records cannot become lineage commits. Failed attempts remain in the agent run
log, not the committed lineage. Persist them in an attempts directory when they
should inform future variation decisions.
