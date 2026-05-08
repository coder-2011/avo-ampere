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
  and cap build parallelism with conservative `MAX_JOBS=1` and `NVCC_THREADS=1`
  defaults on this 32 GB host.
- Run `avo env` before FA2 installation and inspect `baseline_build`. If
  `torch_cuda` and `nvcc_cuda` have different major versions, PyTorch extension
  setup will fail; fix `CUDA_HOME`/`CUDA_PATH`/`PATH` or use a torch build that
  matches the available toolkit before retrying the baseline.
- On this pod, installing `nvidia-cuda-nvcc==13.0.88`,
  `nvidia-cuda-crt==13.0.88`, `nvidia-nvvm==13.0.88`, and
  `nvidia-cuda-cccl==13.0.85` provides a CUDA 13.0 compiler, CRT, NVVM, and CCCL
  headers under the Python site-packages `nvidia/cu13` root. The baseline and
  candidate extension environments prefer that root when the ambient system
  `CUDA_HOME` points at the incompatible CUDA 12.9 toolkit. They also filter
  ambient `/usr/local/cuda*` include/lib paths and add a cached `libcudart.so`
  link shim for NVIDIA wheels that ship only `libcudart.so.13`.
- Score records can use `--trials N` to collect replicate CUDA-event timings.
  Per-case TFLOPS is computed from the median sample, and the JSON includes
  samples, min, mean, median, and coefficient of variation so noisy runs are
  visible instead of hidden. Score summaries also include benchmark settings,
  the A6000/sm86 target, Python/PyTorch/CUDA versions, and visible GPU
  properties so accepted lineage commits carry enough provenance for later
  comparison.
- FlashAttention-2 v2.8.3 has a device-specific block-size heuristic for sm8x
  that treats sm86/sm89 separately from sm80. For head dimension 128, it chooses
  smaller N-blocks on sm86 in some cases: 64 for causal/no-dropout and 32 for
  non-causal/no-dropout. This is useful search-space evidence, not a commandment.
- Local candidates should currently start from
  `candidates/cuda_mma_attention_seed.py` for 16/32-token, head-dim 16
  BF16 tensor-core QK/PV smokes
  and `candidates/cuda_warp_rows_attention_seed.py` for tiny warp-row online-softmax
  scoring. `cuda_tiled_attention_seed.py` is the one-CTA-per-row tiled reference.
  `cuda_naive_attention_seed.py` is the simpler one-thread-per-row attention
  reference. The older `cuda_identity_seed.py` is only an extension/build smoke
  because it delegates attention math to PyTorch SDPA before running a copy
  kernel.
- Use `evolve-once --attempts-dir ./attempts` for a single autonomous decision
  and `evolve-loop --attempts-dir ./attempts --max-steps N` for a bounded
  multi-step session. The loop repeats the same one-step decision/score/gate
  unit, requires an attempts directory for cross-step memory, stops on accepted
  candidates, rejected-patch cleanup failure, or max-step exhaustion, and lets
  command failures or gate rejections inform the next prompt instead of becoming
  committed lineage. Attempt-history summaries include a supervisor signal after
  repeated unaccepted command/edit fingerprints or five unaccepted attempts in a
  row; treat that as a prompt to change strategy, not as permission to expand the
  command allowlist or bypass the gate.
- Candidate source patching now has a bounded manual substrate:
  `avo apply-patch PATCH --dry-run` accepts only ordinary unified diffs under
  `candidates/`, rejects path traversal, symlink-mode patches, binary patches,
  renames, deletes, and mode changes, and runs `git apply --recount --check`
  before any real apply. Anthropic decisions also include `candidate_patch`: a
  non-empty patch is applied through the same validator before the bounded
  `next_command` runs. Planner prompts include bounded excerpts of current
  candidate sources so raw diffs can use real file context. This does not stage,
  commit, score, or expand the `next_command` allowlist beyond `avo env`,
  `avo compile`, and `avo score`. If `evolve-once` applies a patch and the step
  is not accepted, it attempts a checked reverse apply so rejected edits do not
  leak into later attempts. If a candidate step is accepted, the lineage commit
  stores snapshots of the scored candidate module and companion source directory
  under `sources/latest/`; patched accepted steps also store the raw patch under
  `patches/latest.patch`.

## Search Space

- `cp.async` pipeline depth and staging.
- A first inline `cp.async.cg.shared.global` staging attempt on the warp-row seed compiled to
  `LDGSTS.E.BYPASS.128` on sm86 and passed correctness, but regressed the tiny 64x64 BF16
  smoke versus synchronous shared staging. Do not re-enable a single-stage cp.async copy without
  adding actual overlap, double buffering, or profiler evidence.
- A later generated cp.async patch for the warp-row shared K/V staging path was rejected before
  compile because the raw diff had trailing whitespace and corrupt hunk structure. Its proposed
  structure was also invalid for future attempts: it issued 16-byte async copies at scalar element
  positions, dropped zero-fill for out-of-tile lanes, and waited immediately with no double-buffered
  overlap. Any new cp.async patch must use vector-aligned 16-byte groups, preserve zero-fill or
  guarded shared-memory state for partial tiles, and introduce a real overlapped pipeline rather
  than a single-stage copy/wait replacement.
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
  The wrapper currently caps smoke scoring at sequence length 256 and head
  dimension 128. The accepted 256-token BF16 smoke at `seq_len=256`,
  `head_dim=128`, `total_tokens=1024`, and `num_heads=4` reached
  `0.4933314507556887` TFLOPS noncausal, `0.3264049909158355` TFLOPS causal,
  and `0.4012802607933843` geomean TFLOPS. This is the current best lineage
  score. It is still far from FA2: it does not use `mma.sync` or
  `cp.async`. With `--ptxas-options=-v`, the current warp-row kernel reports no
  spills; BF16/Half entry points use 48 registers, 1 barrier, and 16896 bytes
  shared memory, while the FP32 entry point uses 56 registers, 1 barrier, and
  33280 bytes shared memory.
- The tiled seed compiles cleanly on sm86 with no spills. Its ptxas diagnostics
  report 40 registers and 1 barrier for BF16/Half/FP32 entry points, and 48
  registers and 1 barrier for the double entry point. The ptxas output did not
  report static shared-memory allocation for those entry points. It passes the
  tiny BF16 smoke at `seq_len=16`, `head_dim=16`, `total_tokens=16`, and
  `num_heads=1` with geomean `1.1994290536675978e-05` TFLOPS, but fails the
  larger `seq_len=128`, `head_dim=128`, `total_tokens=512`, `num_heads=4` smoke
  with max_abs_error `0.485504150390625` noncausal and `1.4482421875` causal.
  Do not score larger tiled shapes without a patch that fixes or extends the
  tiled kernel first.
- The naive seed is useful only as a correctness reference. A no-patch BF16
  score at `seq_len=128`, `head_dim=128`, `total_tokens=512`, and `num_heads=4`
  passed both causal modes, but it was much slower than the warp-row best:
  noncausal `0.0010531516927932967` TFLOPS, causal `0.000930790492895549`
  TFLOPS, geomean `0.000990082614345315` TFLOPS. The gate rejected this versus
  the then-current `0.10830947571120902` best, so do not repeat naive no-patch
  scoring as a candidate-improving step.
- NVIDIA's CUTLASS CuTeDSL Ampere FlashAttention v2 example is useful search
  evidence for the direction from the warp-row seed toward FA2-like structure:
  it combines 128-bit `cp.async` Q/K/V global-to-shared copies, Ampere BF16/FP16
  tensor-core MMA via `MmaF16BF16Op(..., (16, 8, 16))`, register pipelining for
  shared-to-register copies, online softmax with output rescaling, and head-dim
  padding to multiples of 32. It uses default 128x128 m/n tiles with 128 threads,
  but smaller smoke tiles are still appropriate here until correctness is stable.
- Dao-AILab's CuTe FlashAttention forward path gives concrete guardrails for
  Ampere-style FA2 edits: dtype must be FP16 or BF16, head dimensions and V head
  dimensions are expected to be multiples of 8 for 16-byte alignment, tile-N must
  be divisible by 16, thread count must be a multiple of 32, and shared-memory
  use is budgeted as Q plus staged K/V tiles. For local CUDA patches, keep these
  constraints explicit and prefer small, testable steps such as widening the
  MMA seed shape or adding correctly overlapped staging over another shape-only
  score of the warp-row seed.
- The tiny MMA seed uses CUDA WMMA on sm86 to compute 16x16 BF16 QK score tiles
  and BF16 PV output tiles with tensor cores. It stores unnormalized softmax
  probabilities as BF16 between the two MMA operations and keeps FP32 online
  row-max, row-sum, and output accumulators across up to two key tiles. It is only
  a correctness foothold for tensor-core attention: the wrapper currently accepts
  only sequence lengths 16 or 32 with head dimension 16, and it does not yet use
  production layouts. Do not score head dimension 32 or larger unless the
  candidate patch updates the wrapper and CUDA kernel to support that shape first.
  A no-patch score at the maximum supported smoke shape (`seq_len=32`,
  `head_dim=16`, `total_tokens=32`, `num_heads=1`, BF16, both causal modes)
  passed correctness but was gate-rejected at `6.539498372773744e-05` geomean
  TFLOPS versus the then-current `0.10830947571120902` best, so do not repeat that
  baseline score as a candidate-improving step.
  A patched attempt that simply changed `kHeadDim` and `SMOKE_HEAD_DIM` from 16
  to 32 applied cleanly, but failed CUDA compilation: WMMA fragments such as
  `fragment<matrix_a, 16, 16, 32, ...>` and `fragment<accumulator, 16, 32, 16,
  ...>` are incomplete/unsupported. Any head-dimension-32 MMA extension must keep
  WMMA K fragments at 16 and explicitly process two 16-wide chunks for QK and PV;
  do not repeat the constant-only `kHeadDim=32` patch.
  A follow-up two-chunk generated patch applied and ran the score command, but
  failed CUDA compilation because it only partially replaced `kHeadDim`: stale
  `linear / kHeadDim` row calculations remained, and the output write path
  redeclared `row` after adding `linear / head_dim`. A correct two-chunk patch
  must consistently separate score tile size (`16x16`) from output tile size
  (`16xhead_dim`), size `pv_tile`/`output_acc` for the maximum supported head
  dimension, use `head_dim` for runtime row/dim indexing and global strides, and
  avoid duplicate local declarations in the final store loop.
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
- NVIDIA CUTLASS CuTeDSL Ampere FlashAttention v2 example:
  https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py
- Dao-AILab CuTe FlashAttention forward implementation:
  https://github.com/Dao-AILab/flash-attention/blob/58fe37fb/flash_attn/cute/flash_fwd.py

## Gate

A candidate enters lineage only when it passes correctness and matches or
improves the current best aggregate score. The score payload must also contain
at least one scored case and a finite positive geomean, so malformed empty score
records cannot become lineage commits. Failed attempts remain in the agent run
log, not the committed lineage. Persist them in an attempts directory when they
should inform future variation decisions.
