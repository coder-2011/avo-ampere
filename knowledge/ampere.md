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
- Do not put `flash-attn` directly in the `baseline` optional dependency group:
  `uv run --extra baseline ...` resolves dependencies before AVO can apply the
  sm86/Ampere build environment, so upstream setup falls back to its broad
  default arch list. Install FA2 explicitly with the pinned environment, then
  run `seed-baseline` from the existing CUDA environment.
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
  `candidates/cuda_mma_attention_seed.py` for the accepted seq32768 head-dim 128
  BF16 tensor-core QK/PV lane
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
- A later MMA single-stage cp.async patch was also rejected before compile with
  `error: corrupt patch at line 53`. It proposed scalar BF16 element
  `__pipeline_memcpy_async` calls into a 16x16 K tile and an immediate
  commit/wait before `wmma::load_matrix_sync`, so even a syntactically valid
  version would not be a useful Ampere pipeline. For MMA cp.async attempts, use
  aligned 16-byte groups (8 BF16 elements), keep scalar tails disjoint, and
  compile-check a clean diff before attempting any score.
- A later double-buffered cp.async warp-row patch applied but failed compile on this NVCC/header
  path because `__pipeline_commit` and `__pipeline_wait_prior` were undefined. Do not use those
  intrinsics here unless the necessary CUDA pipeline API/include contract is first proven by a tiny
  compile smoke. Prefer a known-good inline PTX `cp.async` helper or the standard CUDA pipeline API
  with the exact required includes. That patch also treated a BF16 16-byte copy as if it covered
  16 elements; 16 bytes is 8 BF16 elements, so vector groups should be aligned on 8-element
  boundaries. Do not mix scalar fallback stores into the same shared-memory range while an async
  vector copy to that range is pending; handle full 16-byte groups and scalar tails as disjoint
  regions after the wait/commit protocol is correct.
- Local CUDA 13 headers expose the failed `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior` names through `cuda_pipeline_primitives.h`, not through the default
  candidate includes. That header routes 16-byte copies to `cp.async.cg.shared.global`, supports a
  source-size/zero-fill argument for partial copies, and asserts shared/global address spaces plus
  4/8/16-byte alignment. A future cp.async attempt can first add `#include <cuda_pipeline_primitives.h>`
  and compile a tiny candidate-local smoke before restructuring the warp-row loop.
  NVIDIA's current CUDA Programming Guide also documents the primitives as
  function-style calls: `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior(N)`. The higher-level `cuda::pipeline` flow uses
  producer acquire, `cuda::memcpy_async`, producer commit, and consumer wait.
  On Ampere+, `cuda::memcpy_async` can lower to `cp.async` for aligned
  global-to-shared copies. CCCL documents 4-byte alignment as the minimum
  Ampere+ lowering condition, while NVIDIA Ampere material calls out 16-byte
  size/alignment as the better async-copy path. For AVO BF16 throughput patches,
  keep treating 16-byte groups as the target and reject scalar 2-byte async-copy
  noise; 4-byte copies are only useful for API smokes or carefully justified tails.
  Do not use a templated public spelling for `__pipeline_wait_prior`.
  That tiny compile smoke has now succeeded on the warp-row source for sm86: adding the header plus
  unused wrappers around `__pipeline_memcpy_async`, `__pipeline_commit`, and
  `__pipeline_wait_prior` compiled with no spills. NVCC warned only that the commit/wait wrappers
  were unused. This proves header/API availability, not performance. The next cp.async attempt must
  still add a real double-buffered overlap and must keep 16-byte groups aligned and disjoint from
  scalar tail writes.
  A fresh primary-source pass over NVIDIA CUTLASS's Ampere FlashAttention v2 example and
  FlashAttention-2 SM80 traits reinforced the same target shape: Q/K/V global-to-shared copies use
  128-bit copy atoms, BF16 head-dim128 maps to 8 BF16 values per global-copy vector, shared layouts
  are swizzled to manage bank conflicts, and the async path is coupled to tensor-core MMA plus
  online-softmax rescaling. This is not support for scalar BF16 `__pipeline_memcpy_async`; it is
  evidence that useful async-copy patches should be vector-group dataflow changes.
  A first double-buffered cp.async structural patch applied but failed compile because the doubled
  static K/V shared-memory buffers made the FP32 template instantiation use 66048 bytes of shared
  memory, above the 49152-byte static allocation limit. BF16/Half reached ptxas with 33280 bytes
  shared memory, 56 registers, and no spills, but the translation unit still fails while the FP32
  entry point is instantiated. Future double-buffering must either avoid doubling the FP32 static
  buffers, reduce the staged tile footprint, split dtype-specific kernels, or move above-48KB use to
  dynamic shared memory with the required launch attribute.
  NVIDIA's Ampere tuning guide confirms the general rule for sm86: static shared memory remains
  limited to 48 KB for architectural compatibility, while devices with compute capability 8.6 can
  address up to 99 KB per thread block only through dynamic shared memory with explicit opt-in.
  CUDA's function attributes also constrain the requested dynamic shared memory plus static shared
  memory to the device opt-in limit. Do not propose a static shared-memory allocation above 48 KB.
  A later MMA double-buffer skeleton applied but failed compile. It used
  `__pipeline_wait_prior<1>()`, while the local public CUDA primitive is the
  non-template `__pipeline_wait_prior(prior)` wrapper. It also issued scalar
  BF16 `__pipeline_memcpy_async(..., sizeof(__nv_bfloat16))` copies instead of
  16-byte aligned groups and accidentally removed the opening
  `wmma::fragment<wmma::matrix_a, ...>` declaration line, leaving stray template
  arguments and an undefined `q_frag`. The planner now rejects patches that add
  templated `__pipeline_wait_prior<...>` or scalar BF16 async copies.
  A later warp-row async-copy API proof patch added only wrappers plus an empty
  `async_copy_tile_kv` stub that was not called. It still failed compile after
  introducing a `scalar_t` helper outside the templated kernel context and
  triggering cascading syntax errors. The decision text admitted the patch could
  not affect correctness or throughput; the planner now treats that no-op/stub
  language as self-invalid for non-empty patches.
  A later warp-row V-accumulation unroll patch added only `#pragma unroll` before
  two fixed-trip `key_inner` loops. It compiled cleanly for sm86 with the same
  register and shared-memory counts as the baseline, but the agent stopped after
  `avo compile` and produced no correctness or TFLOPS score. Pragma-only or
  scheduler-only performance patches should run a bounded candidate score instead
  of a compile-only check; the planner now rejects pragma-only compile commands.
  A later tiled reset attempt changed only `cuda_tiled_attention_seed.py` wrapper
  caps and scored seq64/head_dim64 BF16. Both noncausal and causal cases failed
  correctness (`max_abs_error` 0.6171875 and 0.2646484375). Raising tiled wrapper
  caps alone is not a correctness fix; larger tiled scores must include a kernel
  change for the known larger-shape failure, and the planner now rejects
  wrapper-cap-only tiled larger-shape scores.
  A later warp-row WMMA QK skeleton patch was rejected before compile because
  `git apply --check` found trailing whitespace in added lines. Its risk text
  also admitted an early return would break correctness if scored. The planner
  now rejects candidate patches with trailing whitespace in added diff lines and
  treats `would break correctness` as self-invalid language for non-empty patches.
  A WMMA source refresh found that CUTLASS lists SM80+ BF16 TensorOp support and
  WMMA 16x16x16 shapes, while NVIDIA's BF16 WMMA sample stages A/B tiles in
  shared memory, uses 16-byte vectorized copies plus skew/padding to satisfy
  WMMA alignment and bank-conflict constraints, and uses explicit
  `__nv_bfloat16` fragments. Future warp-row WMMA work should first build a
  small aligned shared-memory tile path without early returns or dead score tiles;
  direct global-load skeletons that do not integrate with online softmax are not
  useful progress.
  A later warp-row WMMA QK skeleton used `scalar_t` as the WMMA matrix fragment
  element inside the generic PyTorch-dispatched kernel. NVCC instantiated that
  code for `float`, `c10::Half`, and `c10::BFloat16`, and rejected all matrix A/B
  fragments as incomplete. The planner now rejects `scalar_t` WMMA matrix
  fragments; future WMMA patches must use explicit CUDA WMMA element types, such
  as `__nv_bfloat16`, inside dtype-specific code paths.
  The next accepted warp-row improvement added one padding column to both staged
  K and V shared-memory tiles (`kMaxHeadDim + 1`) to reduce bank conflicts in the
  V accumulation path. It preserved correctness on the seq256/head_dim128 BF16
  suite and improved geomean from 0.4012802607933843 to 0.43185073056556733
  TFLOPS: noncausal 0.60304142909027 TFLOPS, causal 0.3092574481513733 TFLOPS.
  Ptxas for sm86 reports no spills; BF16/Half use 48 registers and 17024 bytes
  shared memory, FP32 uses 56 registers and 33536 bytes shared memory.
  A follow-up standalone `#pragma unroll` patch on the V accumulation loops was
  scored after the skew and regressed badly despite passing correctness: geomean
  0.2576601941393183 TFLOPS, noncausal 0.32998084031335845 TFLOPS, causal
  0.20118978902189197 TFLOPS. Do not repeat pragma-only performance patches;
  the planner now rejects patches whose only added lines are `#pragma unroll`.
  A follow-up dynamic shared-memory double-buffer skeleton moved K/V tiles to flat
  `scalar_t*` buffers but did not update the existing 2D `[key][dim]` access sites,
  so compile failed with `no operator []` errors. Its own risk text said the
  doubled buffers were unused, could not improve throughput, and indexing had to
  be updated before scoring; the planner now treats those phrases as self-invalid.
  A corrected dynamic-shared migration then compiled cleanly by keeping only
  `score_tiles` static, moving K/V tiles to flat `extern __shared__` buffers, and
  replacing 2D K/V accesses with `key * (kMaxHeadDim + 1) + dim` indexing. Ptxas
  reported 512 bytes static shared memory, no spills, 48 registers for BF16/Half,
  and 56 registers for FP32. This was compile-only and cleaned up; before adding
  double buffering or `cp.async`, rerun that flat dynamic K/V migration as a
  bounded score on the current seq256/head_dim128 BF16 suite to prove correctness
  and throughput.
  The scored standalone dynamic-shared K/V migration preserved correctness but
  regressed the current best: geomean 0.32420366797036887 TFLOPS, noncausal
  0.42179242571503833 TFLOPS, causal 0.24919370741961078 TFLOPS. Do not repeat
  dynamic-shared K/V migration by itself; only revisit dynamic shared memory when
  adding real async-copy or double-buffering logic that can offset the overhead.
  A later warp-row patch converted the shared `score_tiles` staging buffer from
  FP32 to BF16 to halve that buffer's shared-memory footprint. It preserved
  correctness on the fixed seq256/head_dim128 BF16 suite, but regressed the
  shared-memory skew best: geomean `0.3565090838055145` TFLOPS versus
  `0.43185073056556733`, with noncausal `0.44378300977557367` TFLOPS and causal
  `0.28639836144273007` TFLOPS. Do not repeat the BF16 `score_tiles`
  conversion as a candidate-improving step; the planner now rejects that exact
  buffer-precision change.
  A direct threshold-only change from `head_dim <= 64` to `head_dim <= 128` for
  `can_stage_shared` is unsafe. The generated patch was corrupt, and a manual
  one-line score check failed correctness: noncausal hit a CUDA unknown error and
  causal hit a misaligned-address error. Keep the threshold at 64 unless a patch
  also fixes the head_dim128 shared-path alignment/layout issue; the planner now
  rejects direct threshold-only changes to 128.
  The likely local cause is the interaction between the `kMaxHeadDim + 1` skew
  and the vectorized `dot_product`: BF16 row stride becomes 129 elements / 258
  bytes, so most rows are not aligned for the `ScalarPack4` reinterpret path.
  NVIDIA/CUTLASS guidance also emphasizes 16-byte alignment for vectorized shared
  memory IO. A future head_dim128 shared path should use a stride that preserves
  pack alignment (for example a multiple of 4 BF16 elements for the current
  `ScalarPack4`, or 8 BF16 elements for 16-byte alignment) or explicitly fall
  back to scalar loads before enabling shared staging at head_dim128.
  Do not change `kTileKeys` above `kWarpSize` in the warp-row kernel unless the score and V
  accumulation loops are also changed to map multiple key columns per lane. With the current
  one-key-per-lane mapping, `kTileKeys=64` makes 32 lanes process only keys 0..31 while advancing
  the tile by 64, skipping half the keys and breaking correctness.
  A dynamic shared-memory structural patch that moved K/V staging to `extern __shared__` compiled on
  sm86 with only 512 bytes of static shared memory, no spills, and the usual 48/56 registers for
  Half/BF16/FP32. It is not scoreable as generated: the kernel launch did not pass the required
  dynamic shared-memory byte count, the opt-in attribute was set only for the BF16 specialization,
  and the dynamic byte count used BF16 size unconditionally. Future dynamic-shared patches must wire
  both the launch third argument and dtype-specific `cudaFuncSetAttribute` values before scoring.
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
  A shape-only wrapper patch to score `seq_len=512`, `head_dim=128`,
  `total_tokens=2048`, and `num_heads=4` also passed correctness with
  `0.8677167693061046` geomean TFLOPS, but the fixed-case gate rejected it
  because the benchmark signature differed from the seq256 best. Do not treat
  larger-shape TFLOPS as a gate improvement unless the benchmark suite is
  deliberately reseeded.
  A later no-edit rerun of the exact same seq256/head_dim128 warp-row source
  measured faster (`0.460232249967343` geomean TFLOPS) and was briefly accepted
  due timing noise. The lineage gate now rejects unchanged source snapshots when
  no candidate patch is present, so identical-source reruns cannot advance the
  lineage solely by sampling a faster timing.
  A compile-first patched warp-row WMMA attempt applied and compiled on sm86
  with no spills: BF16 used 48 registers, 1 barrier, and 17984 bytes shared
  memory; FP16 used 48 registers and 16896 bytes shared memory; FP32 used 56
  registers and 33280 bytes shared memory. The patch is not scoreable as-is: it
  only handles a `head_dim == 16` BF16 score path for `warp_id == 0`, does not
  integrate the WMMA scores into the existing online softmax/output accumulation
  for all rows, and would leave that branch without a final output update.
  A later warp-row single-tile WMMA probe added `mma.h` and fragment declarations
  inside the warp-row kernel and compiled with unchanged BF16/Half diagnostics
  (48 registers, 1 barrier, 16896 bytes shared memory, no spills; FP32 56
  registers, 33280 bytes shared memory). It is compile-only and not scoreable:
  the generated code gated WMMA work behind `lane == 0`, did not store the WMMA
  score fragment into the existing `scores` array, and would be skipped on the
  head_dim128 target because `can_stage_shared` is false. NVIDIA's warp-level
  primitive guidance requires all participating warp threads to execute
  collectives coherently; future WMMA work must not call `load_matrix_sync` or
  `mma_sync` from a single lane.
  A later scalar fallback unroll patch was rejected before compile because its
  context did not match the current source and it referenced `qv`, `kv`, and
  `inner` outside the packed branch where those names are defined. The fixed
  benchmark head dimension is 128, so the current dot-product path takes the
  divisible-by-4 packed branch, not the scalar fallback. Do not spend another
  candidate on scalar fallback unrolling unless the benchmark suite includes an
  odd/non-packed head dimension. NVIDIA CUTLASS guidance also frames unrolling
  as most useful for loops with compile-time-known trip counts; the scalar
  fallback loop uses runtime `head_dim`.
  A later q-offset hoist patch was rejected before compile because its context
  did not match the current warp-row source. The source already computes
  `q_offset` before both shared and global K/V tile loops; merely introducing a
  `q_row = q + q_offset` pointer is likely too small to beat timing noise, and
  any such patch must use exact context after `can_stage_shared` is declared if
  it branches on that value.
  A small patch adding `#pragma unroll` to the packed 4-wide dot-product outer
  loop applied and compiled on sm86. BF16/Half diagnostics stayed at 48
  registers, 1 barrier, and 16896 bytes shared memory with no spills; FP32 rose
  to 64 registers and 33280 bytes shared memory with no spills. This is a
  plausible low-risk score candidate on the fixed seq256/head_dim128 BF16
  warp-row suite; the next step can score the same patch rather than repeating
  compile-only.
  That patch was scored with three trials and passed correctness but regressed:
  noncausal `0.4773985262900999` TFLOPS, causal `0.2881296921121001` TFLOPS,
  and `0.3708809652634344` geomean TFLOPS versus the current `0.4012802607933843`
  best. Do not repeat the packed dot-product outer-loop unroll as a
  candidate-improving step.
  NVIDIA's CUDA Programming Guide says the mapping of matrix elements into
  WMMA fragment internal storage is unspecified and can change across
  architectures. Do not infer row/column positions from `fragment.x[]`; apply
  only uniform per-element transforms there, or use `wmma::store_matrix_sync`
  into shared/register-backed memory with an explicit row/column layout before
  consuming selected rows or columns.
  Future warp-row WMMA work should either keep the normal path intact and compile
  an isolated helper, or fully route all rows through a correct online-softmax
  path before scoring.
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
  A generated tiled online-softmax rescale patch was rejected before compile
  because its context did not match the current source. More importantly, its
  own risk text noted that it removed `tile_scale` while leaving a stale
  `row_sum` reference that would cause a compile error. The planner now rejects
  non-empty patches that describe themselves as known invalid before execution.
  Future tiled fixes should keep the online-softmax invariant explicit:
  `output_acc = output_acc * old_scale + tile_acc * tile_scale`, with
  `row_sum = row_sum * old_scale + tile_sum * tile_scale`.
  The current tiled source already contains that exact output recurrence. A
  later loop repeated a stale patch from `output_acc = tile_acc * tile_scale` to
  the correct recurrence, but that old line is no longer present and `git apply`
  rejected the patch. The planner now rejects this stale tiled rescale fix; look
  for other larger-shape correctness causes instead.
  Dao-AILab's CuTe softmax helper reinforces this recurrence: it computes a
  per-row scale from the previous row max to the current row max, updates the
  running row sum using `old_row_sum * row_scale` as the reduction initializer,
  and exposes `rescale_O` to multiply the accumulated output by that row scale
  before consuming the current probability tile. Do not remove either the old
  accumulator scale or the current tile scale when the local probabilities were
  normalized to a tile-local maximum.
  A later tiled rescale probe did the opposite: it removed `tile_scale` from the
  output update while its own risk text said the patch would break correctness
  and should be rejected. That patch applied and compiled, then was cleaned up
  because it was compile-only. The planner now also rejects non-empty patches
  whose decision text says either `will break correctness` or
  `reject this direction`.
  A later no-edit run repeated the already-known tiny tiled smoke at
  `seq_len=16`, `head_dim=16`, `total_tokens=16`, and `num_heads=1`. It passed
  correctness with geomean `1.5149841720005358e-05` TFLOPS but was gate-rejected
  because its case signature differs from the current warp-row best. The planner
  now rejects unpatched repeats of that tiled smoke; future tiled scores need a
  `candidate_patch` that fixes or extends the kernel/wrapper.
  A later tiled reduction-bound guard patch initialized out-of-tile reduction
  lanes to `-inf` for max and `0.0f` for sum, but it still failed the
  seq128/head_dim128 BF16 correctness check: max_abs_error `0.672119140625`
  noncausal and `0.927734375` causal, with geomean zero. Do not repeat that
  exact `reduce[tid] = score/shifted` guard patch; the planner now rejects it.
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
  only sequence lengths 16 or 32 with head dimension 64, and it does not yet use
  production layouts. Do not score larger shapes unless the candidate patch
  updates the wrapper and CUDA kernel to support that shape first.
  A no-patch score at the maximum supported smoke shape (`seq_len=32`,
  `head_dim=16`, `total_tokens=32`, `num_heads=1`, BF16, both causal modes)
  passed correctness but was gate-rejected at `6.539498372773744e-05` geomean
  TFLOPS versus the then-current `0.10830947571120902` best, so do not repeat that
  baseline score as a candidate-improving step. A later no-edit diagnostic
  repeated the same smoke after the fixed-case gate and again passed correctness
  at `7.894282511202798e-05` geomean TFLOPS, but was rejected because its case
  signature differs from the seq256 warp-row best. Do not spend another loop on
  the unpatched MMA baseline unless checking that the CUDA extension toolchain
  still works after an environment change.
  A subsequent no-edit compile diagnostic for
  `candidates/cuda_mma_attention/attention_kernel.cu` succeeded on sm86 with 40
  registers, 1 barrier, and 3776 bytes of shared memory, but it did not test a
  source change or score a candidate. The planner/validator now blocks repeated
  no-patch compiles of that recorded MMA source; future MMA compiles should
  build-check a non-empty candidate patch.
  A small patch adding `#pragma unroll` before several MMA seed helper loops
  applied and compiled on sm86 with no spills and unchanged diagnostics:
  40 registers, 1 barrier, and 3776 bytes shared memory. This is not a useful
  lineage candidate by itself because the MMA wrapper is still limited to the
  tiny seq32/head_dim16 case signature, which differs from the current seq256/
  head_dim128 warp-row best. Do not repeat this as another compile-only step;
  any future MMA unroll score should be paired with a deliberately reseeded MMA
  benchmark or a wrapper/kernel extension that can compete on the target suite.
  A later head-dimension-32 MMA attempt proposed the right broad direction
  (two 16-wide QK chunks, widened `pv_tile`/`output_acc`, and PV stores at
  `&pv_tile[chunk * 16]`), but the candidate patch was rejected before compile:
  `git apply --check` reported trailing whitespace and a corrupt hunk. Future
  MMA patches should be smaller, use exact current file context, avoid
  whitespace-only added lines, and compile-check the first structural slice
  before bundling QK, PV, wrapper, and score-shape changes together.
  A second generated head-dimension-32 patch again tried to bundle wrapper,
  QK, PV, and scoring into one step and was rejected by `git apply --check`
  with `error: corrupt patch at line 120`. It also introduced undefined
  `head_dim` identifiers inside the CUDA kernel and still used unsupported
  WMMA fragment template shapes. The validator now rejects patched MMA scores
  beyond the current seq256/head_dim128 smoke unless the next command is a
  compile build-check first.
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
  A later two-chunk patch compiled and ran, but failed tolerance badly
  (`max_abs_error` about `86.64` noncausal and `218.83` causal). That patch
  widened `pv_tile` to `kTile*kHeadDim` but left `output_acc` at `kTileElements`
  while loops wrote `kTile*head_dim`, likely corrupting shared memory. The next
  correctness attempt must widen both `pv_tile` and `output_acc`, keep score and
  probability tiles at `16x16`, and verify the PV store offsets for each
  16-wide output chunk.
  Do not use `linear / kTile` to compute output rows in widened 16x32 loops; the
  row stride is the head dimension, so row indexing should use `linear / head_dim`
  or `linear / kHeadDim` when `kHeadDim` is the runtime stride. A later generated
  patch with `linear / kTile` was rejected before compile due corrupt diff
  structure, but the indexing direction itself was also wrong.
  A subsequent two-chunk patch compiled and ran with widened `pv_tile` and
  `output_acc`, but still failed tolerance (`max_abs_error` about `1.748`
  noncausal and `2.547` causal). That patch stored the second 16-wide PV output
  chunk at `&pv_tile[chunk * kTile * 16]` while using a row stride of
  `kHeadDim == 32`, which treats chunk 1 as a later row block and writes past the
  16x32 tile. For row-major widened output tiles, each 16-wide PV chunk should
  store at the column offset, e.g. `&pv_tile[chunk * 16]`, with leading dimension
  `kHeadDim`.
  A compile-only two-chunk MMA structural patch later applied and compiled on
  sm86 with no spills: ptxas reported 40 registers, 1 barrier, and 3776 bytes
  shared memory. This proves the two 16-wide WMMA fragment loops can compile, but
  the patch is not scoreable or correctness-proven. It kept `kHeadDim == 16` and
  the 16x16 `pv_tile`/`output_acc` buffers, loaded chunk 1 with `kHeadDim` as the
  global row stride, and stored PV chunk 1 with leading dimension `kHeadDim * 2`
  into buffers sized only for 16x16. A real head_dim32 patch must introduce a
  runtime or constant 32-wide row stride, widen `pv_tile` and `output_acc` to
  16x32, and store each 16-column PV chunk at a column offset that stays inside
  the widened row-major tile before running a score.
  A follow-up internal-32 compile patch widened `pv_tile` and `output_acc`, but
  failed CUDA compilation because it left stale original QK code after the new
  two-chunk loop: `score_frag`, `q_frag`, and `k_frag` were referenced after
  their declarations had been replaced by scoped chunk-local fragments. When
  rewriting the MMA QK block, remove the old single-fragment fill/load/mma lines
  completely and store the accumulated two-chunk score fragment only after the
  chunk loop.
  Another generated head_dim32 two-chunk patch was rejected before compile
  because its context did not match the current source and it again left stale
  single-chunk QK load/mma lines after the new two-chunk loop. The decision text
  itself warned those stale lines might reference undeclared fragments and must
  be removed. The planner now rejects non-empty patches when their own risk text
  calls out stale code that still needs removal or may reference undeclared
  symbols.
  Another generated head_dim32 two-chunk patch kept WMMA K at 16, but was still
  rejected by `git apply --check` after it left old single-chunk PV fragment
  lines after the new two-chunk PV loop. Its own risk text said incomplete
  removal of old single-chunk lines was the main risk and that those lines should
  be completely removed. The planner now treats that incomplete-removal warning
  as self-invalid for non-empty patches.
  A subsequent v3 generated patch was again rejected by `git apply --check` and
  left an orphan post-QK `wmma::fragment<wmma::matrix_b, ...> k_frag;` block
  after storing the score tile. The planner now rejects that exact orphan
  post-score-store `k_frag` pattern; a valid two-chunk QK rewrite must remove all
  old single-chunk fragment declarations.
  A later head_dim32 two-chunk patch removed the stale single-chunk QK lines and
  widened `pv_tile`/`output_acc`, but still declared the score accumulator as
  `wmma::fragment<wmma::accumulator, kTile, kTile, kHeadDim, float>` after
  setting `kHeadDim = 32`. NVCC rejected the instantiated
  `fragment<accumulator, 16, 16, 32, float>` as incomplete/unsupported. The
  planner now rejects this specific score-fragment shape; future two-chunk QK
  patches must keep each WMMA fragment K at 16 and accumulate the two chunks
  into a valid 16x16 score accumulator.
  A manual head_dim32 two-chunk MMA seed extension was committed after those
  guardrails. The current MMA seed keeps the score and probability tiles at
  16x16, widens `pv_tile` and `output_acc` to 16x32, processes QK and PV as two
  16-wide WMMA chunks with explicit `__nv_bfloat16` fragments, and stores each
  PV chunk at the row-major column offset `&pv_tile[chunk * 16]` with leading
  dimension 32. The wrapper now accepts only sequence lengths 16 or 32 with
  head dimension 32, total tokens up to 32, and one head. The sm86 compile
  check succeeded with no spills, 40 registers, 1 barrier, 5824 bytes shared
  memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory.
  A fresh tiny score at seq_len 32, total_tokens 32, num_heads 1, head_dim 32,
  BF16, both causal modes passed correctness with max_abs_error 0.00390625 in
  both modes and geomean `0.0001245601243057133` TFLOPS. Noncausal median was
  0.74099200963974 ms / `0.00017688719756063954` TFLOPS; causal median was
  0.7471680045127869 ms / `8.771253533900277e-05` TFLOPS. This is structural
  correctness progress only: the workload signature differs from the current
  seq256/head_dim128 warp-row best and must not be used as a lineage-speed
  comparison.
  A subsequent generated head_dim64 four-chunk MMA patch applied and compiled,
  then was cleaned up because it was compile-only. A manual version of that same
  structural step was committed with the runtime check message fixed. The
  current MMA seed uses `kHeadDim = 64`, widens `pv_tile` and `output_acc` to
  16x64, and processes QK/PV as four 16-wide WMMA chunks while keeping the score
  and probability tiles at 16x16. The sm86 compile check succeeded with no
  spills, 40 registers, 1 barrier, 9920 bytes shared memory, 400 bytes
  `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global memory. A fresh tiny score
  at seq_len 32, total_tokens 32, num_heads 1, head_dim 64, BF16, both causal
  modes passed correctness with max_abs_error 0.00390625 in both modes and
  geomean `0.0003318536197406504` TFLOPS. Noncausal median was
  0.6331200003623962 ms / `0.00041405104853732224` TFLOPS; causal median was
  0.4927999973297119 ms / `0.0002659740274152339` TFLOPS. This is still
  structural correctness progress only because the workload differs from the
  seq256/head_dim128 warp-row best.
  A later generated head_dim128 MMA patch applied and compiled on sm86 with no
  spills, 40 registers, 1 barrier, and 18112 bytes shared memory, but it was
  self-invalid: it changed `kHeadDim` and `SMOKE_HEAD_DIM` to 128 while leaving
  the QK/PV chunk loops at four 16-wide chunks, so it covered only 64 of 128
  dimensions. The decision risk explicitly said it would fail correctness if
  scored. Cleanup succeeded. The planner now rejects that partial head_dim128
  pattern and any patch whose own decision text says it will fail correctness.
  A subsequent generated head_dim128 eight-chunk MMA patch applied and compiled,
  then was cleaned up because it was compile-only. A manual version was
  committed after scoring. The current MMA seed uses `kHeadDim = 128`, widens
  `pv_tile` and `output_acc` to 16x128, and processes QK/PV as eight 16-wide
  WMMA chunks while keeping score and probability tiles at 16x16. The sm86
  compile check succeeded with no spills, 40 registers, 1 barrier, 18112 bytes
  shared memory, 400 bytes `cmem[0]`, 224 bytes `cmem[4]`, and 28 bytes global
  memory. A fresh tiny score at seq_len 32, total_tokens 32, num_heads 1,
  head_dim 128, BF16, both causal modes passed correctness. Noncausal max error
  was 0.00390625 with median 0.9678720235824585 ms /
  `0.0005416914501355385` TFLOPS; causal max error was 0.001953125 with median
  0.9702720046043396 ms / `0.0002701757844769497` TFLOPS. Geomean was
  `0.00038255968486606857` TFLOPS. This is structural correctness progress, not
  a lineage-speed comparison. The planner now treats head_dim128 as the current
  unpatched MMA smoke cap and requires compile-first validation for patched MMA
  scores beyond that cap.
  A subsequent seq64 MMA patch changed only the smoke sequence cap and
  `kMaxSeqLen` from 32 to 64. It passed correctness at seq_len 64, total_tokens
  256, num_heads 4, head_dim 128, BF16, both causal modes, but was gate-rejected
  because the case signature still differs from the seq256/head_dim128 warp-row
  best. A fresh committed-tree score also passed correctness. Noncausal max
  error was 0.00390625 with median 0.6896960139274597 ms /
  `0.04865104527562075` TFLOPS; causal max error was 0.0078125 with median
  0.6171200275421143 ms / `0.027186309390769315` TFLOPS. Geomean was
  `0.03636815047603261` TFLOPS. The current MMA seed now supports
  seq_lens 16/32/64 with head_dim128, total_tokens up to 256, and num_heads up
  to 4. Do not repeat the exact no-patch seq64 score as a candidate-improving
  step.
  A later generated seq128 MMA patch changed only `kMaxSeqLen` and the smoke
  sequence cap from 64 to 128. It compiled on sm86 with no spills, 40 registers,
  1 barrier, and 18112 bytes shared memory, then was manually scored from the
  committed tree at seq_len 128, total_tokens 512, num_heads 4, head_dim 128,
  BF16, both causal modes. Correctness passed. Noncausal max error was
  0.00390625 with median 0.8828799724578857 ms /
  `0.1520226216326391` TFLOPS; causal max error was 0.0078125 with median
  0.7715200185775757 ms / `0.08698266070104863` TFLOPS. Geomean was
  `0.1149927481033293` TFLOPS. This extended the MMA seed to seq_lens
  16/32/64/128 with head_dim128, total_tokens up to 512, and num_heads up to 4.
  Do not repeat the exact no-patch seq128 score as a candidate-improving step.
  A later generated seq256 MMA patch changed only `kMaxSeqLen` and the smoke
  sequence cap from 128 to 256. The generated runtime check dropped seq32, so
  the manual version used an explicit 16/32/64/128/256 check. It compiled on
  sm86 with no spills, 40 registers, 1 barrier, and the same 18112 bytes shared
  memory. A fresh lineage score at seq_len 256, total_tokens 1024, num_heads 4,
  head_dim 128, BF16, both causal modes passed correctness and was accepted over
  the prior seq256 warp-row best. Noncausal max error was 0.001953125 with
  median 0.7603840231895447 ms / `0.7060523309629976` TFLOPS; causal max error
  was 0.0078125 with median 0.7816960215568542 ms /
  `0.3434013332514782` TFLOPS. Geomean was `0.4924015757468769` TFLOPS versus
  the prior best `0.43185073056556733`. The current MMA seed now supports
  seq_lens 16/32/64/128/256 with head_dim128, total_tokens up to 1024, and
  num_heads up to 4. Do not repeat the exact no-patch seq256 score as a
  candidate-improving step; future MMA work should change the kernel structure
  or compile-check a shape extension beyond seq256 before scoring.
  A follow-up Anthropic loop proposed a synchronous shared-memory K staging
  substrate for only QK chunk 0: add `k_shared[kTile * kHeadDim]`, load the
  full 16x128 BF16 K tile cooperatively, and use shared memory only for the
  first 16-wide WMMA K fragment while leaving the other seven QK chunks and all
  PV chunks on global loads. The patch compiled on sm86 with no spills, 40
  registers, 1 barrier, and 22208 bytes shared memory, then was cleaned up
  because it was compile-only. The agent's risk text understated the smem
  increase: 16*128 BF16 values add 4096 bytes, matching the compile delta from
  18112 to 22208 bytes. Do not repeat the exact single-chunk synchronous K
  staging compile-only probe; the next K/V staging step should either stage all
  QK chunks and score, or introduce a clearly different cp.async/double-buffered
  load path and compile-check it first.
  The next loop proposed full synchronous K staging and compiled, but its own
  risk text identified a tile-local addressing bug: it loaded K fragments from
  `k_shared + key_start * kHeadDim + chunk_offset`. Because `k_shared` stores
  only the current 16x128 K tile, the WMMA load base should be
  `k_shared + chunk_offset` with leading dimension `kHeadDim`. The bad patch
  compiled with the same 40 registers, 1 barrier, and 22208 bytes shared
  memory, then was cleaned up without score or gate decision. The planner now
  rejects this global-offset shared-K pattern before compile.
  A corrected manual full-K staging patch used tile-local
  `k_shared + chunk_offset`, compiled with the same diagnostics, and passed
  correctness on the seq256/head_dim128 BF16 suite, but the lineage gate
  rejected it for throughput regression: geomean `0.30611777431945414` TFLOPS
  versus best `0.4924015757468769`. Noncausal was `0.4325250932830868` TFLOPS
  at median 1.2412480115890503 ms; causal was `0.2166535380479405` TFLOPS at
  median 1.2390079498291016 ms. Do not repeat synchronous static `k_shared`
  staging without real async-copy or double-buffered overlap.
  A later planner loop still retried static synchronous K staging after the no-patch and scalar-async
  guards. Runtime feedback now explicitly tells the agent to stop retrying `k_shared` K staging and
  either add real overlap or choose a different non-K-staging patch.
  A later manual Q-staging score tested the analogous static `q_shared[kTile * kHeadDim]` path.
  It compiled with no spills, 40 registers, 1 barrier, and 22208 bytes shared memory, then passed
  correctness but regressed to geomean `0.42035719120740594` TFLOPS. Noncausal was
  `0.5929811502224299` TFLOPS at median 0.9053760170936584 ms; causal was
  `0.2979861470023096` TFLOPS at median 0.9008319973945618 ms. Do not repeat static synchronous
  `q_shared` staging without real overlap or a materially different Q/K dataflow.
  A follow-up loop proposed static double-buffered V staging in the MMA seed:
  `v_shared[2][kTile * kHeadDim]`, cooperative warp loads of the current and
  next 16x128 V tiles, and PV `wmma::load_matrix_sync` from
  `v_shared[current_buffer]`. It compiled with no spills, 40 registers, 1
  barrier, and 26304 bytes shared memory, then passed correctness when manually
  scored. Throughput still regressed to geomean `0.31531656344385717` TFLOPS
  versus best `0.4924015757468769`: noncausal `0.4066415256706702` TFLOPS at
  median 1.320255994796753 ms, causal `0.24450167753542637` TFLOPS at median
  1.0978879928588867 ms. Do not repeat static synchronous `v_shared[2]`
  staging; revisit V staging only with real async-copy overlap or a materially
  different scheduling strategy.
  A later single-buffer `v_shared[kTile * kHeadDim]` compile-only variant also compiled at 22208
  bytes shared memory and was cleaned up, but it was the same synchronous V-staging direction without
  overlap. The planner now rejects both single-buffer and double-buffer static V staging repeats.
  A later probability-buffer skew patch tried to pad the 16x16 BF16
  `probabilities` tile before the PV WMMA load. The generated `kTile + 1`
  leading dimension was invalid for WMMA because NVIDIA documents
  `load_matrix_sync` leading dimensions for half-type multiplicands as
  16-byte-aligned strides. A manually corrected `kProbabilityStride = kTile + 8`
  variant compiled cleanly with no spills, 40 registers, 1 barrier, and 18368
  bytes shared memory, and passed correctness. It still regressed throughput:
  geomean `0.46984155560491525` TFLOPS versus best `0.4924015757468769`,
  noncausal `0.6514158963616397` TFLOPS at median 0.8241599798202515 ms, causal
  `0.3388788769297927` TFLOPS at median 0.7921280264854431 ms. Do not repeat
  simple probability-buffer skew padding without profiler evidence or a
  materially different probability/PV dataflow.
  A later MMA async-copy header check recovered from the scalar BF16 async-copy
  retry loop, but still produced only unused wrapper helpers around
  `__pipeline_memcpy_async`, `__pipeline_commit`, and `__pipeline_wait_prior`.
  It inserted those helper definitions inside the `mma_attention_kernel`
  signature and duplicated the kernel declaration, so NVCC failed with parameter
  syntax and undefined helper-argument errors before ptxas. Cleanup succeeded.
  Do not repeat wrapper-only async-copy API proofs; the CUDA pipeline primitive
  header availability is already recorded. If adding wrappers, place helper
  definitions before the kernel signature and call them in real 16-byte-group
  dataflow, otherwise choose a materially different non-async patch.
  A later no-edit MMA seq256/head_dim128 rescore was correctly rejected by the
  lineage gate due to timing regression: geomean `0.3399665809331029` TFLOPS,
  noncausal `0.4969407540080716` TFLOPS at median 1.0803519487380981 ms, and
  causal `0.23257757633914394` TFLOPS at median 1.1541759967803955 ms. This
  confirms that fresh-baseline no-patch scores waste loop budget and can look
  worse under noise; the planner now rejects unpatched MMA seed scores and
  requires a structural `candidate_patch` before scoring that source again.
  A later loop after Q-staging rejection again failed planning validation on scalar BF16
  `__pipeline_memcpy_async` after three attempts. The base repo context now treats cp.async as a
  cooled-down direction unless the diff is a complete 16-byte-group dataflow change with exact
  current context and no scalar async calls.
  A later non-async MMA register-row-state patch applied and scored, but failed correctness for both
  causal modes with non-finite outputs. It moved `row_max`, `row_sum`, and `old_scale` into
  per-thread arrays, then used `row / blockDim.x` in output-scaling/final-store loops where threads
  did not own the row state. Do not repeat this per-thread register-state mapping; any register
  row-state patch must keep all row consumers on the owning thread or publish state safely.
  A later QK software-pipeline skeleton added `k_frag_next` and loaded the next key tile's first
  WMMA fragment, but never consumed the preloaded fragment in `mma_sync`. It compiled with unchanged
  resource counts and was cleaned up as a compile-only no-op. Do not repeat unused WMMA preload
  skeletons; wire the preloaded fragment into real QK/PV dataflow before compile-checking.
  A later planner loop failed validation after three attempts because the proposed patch described
  itself as known-bad with "will cause a compile error". Treat self-invalid decision text as a hard
  stop: submit a corrected diff that removes the called-out flaw, or use no-edit diagnostic mode.
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
- NVIDIA CUDA Programming Guide WMMA `load_matrix_sync` requirements:
  https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html
- Dao-AILab CuTe FlashAttention forward implementation:
  https://github.com/Dao-AILab/flash-attention/blob/58fe37fb/flash_attn/cute/flash_fwd.py
- Dao-AILab SM80 FlashAttention mainloop:
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm80.hpp
- Dao-AILab CuTe FlashAttention online softmax helper:
  https://github.com/Dao-AILab/flash-attention/blob/58fe37fb/flash_attn/cute/softmax.py
- NVIDIA CUDA Samples BF16 Tensor Core GEMM:
  https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu
- NVIDIA CUDA C++ Programming Guide, Warp Matrix Functions:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions
- NVIDIA CUDA Programming Guide, Advanced Kernel Programming / async copy primitives:
  https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
- NVIDIA CUDA Programming Guide, Pipelines:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html
- NVIDIA CCCL/libcu++ `cuda::memcpy_async` reference:
  https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html
- NVIDIA GTC 2020 Ampere CUDA architecture presentation:
  https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21170-cuda-on-nvidia-ampere-gpu-architecture-taking-your-algorithms-to-the-next-level-of-performance.pdf
- NVIDIA CUTLASS programming guidelines, loop unrolling:
  https://docs.nvidia.com/cutlass/4.4.0/media/docs/cpp/programming_guidelines.html#loop-unrolling
- NVIDIA CUTLASS functionality tables, WMMA and TensorOp layouts:
  https://docs.nvidia.com/cutlass/4.4.0/media/docs/cpp/functionality.html
- NVIDIA cuBLASDx performance guidance, alignment and shared-memory layouts:
  https://docs.nvidia.com/cuda/archive/13.0.1/cublasdx/performance.html
- vLLM FlashAttention SM80 kernel traits:
  https://github.com/vllm-project/flash-attention/blob/8798f277/csrc/flash_attn/src/kernel_traits.h

## Gate

A candidate enters lineage only when it passes correctness and matches or
improves the best aggregate score for the same benchmark case signature. The
lineage also stores per-signature score lanes, so a larger validation workload
can establish a new benchmark case set instead of being rejected solely because
the current latest commit was scored on a smaller smoke suite. Shape-only changes
still must not win an existing throughput lane by changing the workload; later
candidates on the same signature compare apples-to-apples against that lane's
best score. The score payload must also contain at least one scored case and a
finite positive geomean, so malformed empty score records cannot become lineage
commits. Failed attempts remain in the agent run log, not the committed lineage.
Persist them in an attempts directory when they should inform future variation
decisions.
Agent compile commands should write build artifacts under `build/`, not under
`candidates/`, so rejected compile-only steps do not leave object files beside
source files.
  A later planner loop still retried an invalid compile output path after this
  guard, failing validation after three attempts with `--out-dir must be under:
  build`. Retry feedback now explicitly tells the agent to use `build/<name>` and
  not write compiler artifacts under `candidates/`.
  A later Q-preload patch applied but failed compile because it declared
  `wmma::fragment<wmma::matrix_a, kTile, kTile, 16, wmma::row_major>` without
  the required matrix element type, and its own risk text said "Do not use this
  diff". The planner now treats that language as self-invalid and rejects WMMA
  matrix fragments that omit `__nv_bfloat16` or another supported element type.
  A follow-up QK software-pipeline patch used a valid `k_frag_next` WMMA
  matrix-B fragment, loaded chunk 0 before the QK chunk loop, consumed
  `k_frag_next` in `mma_sync`, and loaded `next_chunk` at the end of each
  chunk. It compiled with unchanged resources and preserved correctness, but
  regressed geomean throughput to `0.4462305013884498` TFLOPS versus the
  current best `0.4924015757468769`. Do not repeat this exact QK
  `k_frag_next` preload chain; future QK scheduling work needs materially
  different overlap or profiler evidence.
  A later loop still failed planning validation after three attempts by trying
  to score `cuda_mma_attention_seed.py` with no `candidate_patch`. Retry
  feedback now explicitly says not to repeat no-edit MMA seed scores; scoring
  that candidate again requires a structural raw diff or a different diagnostic.
  A later PV-preload patch applied but failed compile because it used
  `chunk_offset` after moving the declaration into an `else` block and
  referenced `chunk` outside the PV chunk loop. Its own risk text said it would
  fail compilation and not to score the patch as-is. The planner now treats
  those self-invalid warnings as a hard rejection before patch application.
  A manually corrected PV direct-accumulation patch removed the intermediate
  float `pv_tile` shared-memory buffer. It rescales `output_acc` before the PV
  loop, loads each 16-column `output_acc` chunk into the WMMA accumulator
  fragment, runs the PV MMA, and stores the accumulator directly back to
  `output_acc`. This compiled with no spills, 40 registers, 1 barrier, and
  9920 bytes shared memory. It preserved correctness and improved seq256
  head_dim128 BF16 geomean to `0.5772885607891738` TFLOPS, so lineage accepted
  commit `845ab85`.
  A later post-acceptance loop retried probability-buffer stride-24 padding
  using a 2D `probabilities[kTile][kProbabilityStride]` declaration and
  `&probabilities[0][0]` in the WMMA load. It failed the noncausal build because
  the masked-tile zero-fill branch still used stale flattened
  `probabilities[row * kTile + key]` indexing. This is still the same simple
  probability-buffer skew direction that previously regressed throughput, so the
  planner now rejects both flat and 2D stride-24 repeats.
  A later score-tile skew patch changed `scores` to a padded 2D tile but also
  accidentally changed the probability store to `probabilities[row][key]` while
  leaving `probabilities` flat. NVCC failed with no matching `operator[]`. The
  risk text said the diff was incomplete and should be rejected. The planner now
  treats those self-invalid phrases as hard stops and rejects 2D probability
  indexing unless the diff also declares `probabilities` as a 2D shared tile.
  A later loop failed planning validation after three attempts because the
  decision text still described its patch as unable to improve throughput.
  Retry feedback now explicitly includes "cannot improve throughput" among the
  self-invalid patch descriptions that require a corrected diff or no-edit
  diagnostic instead of another patch attempt.
  A later corrected score-tile skew patch padded `scores` with
  `kScoreStride = 24`, updated the WMMA score store to `&scores[0][0]`, and
  used 2D score indexing. It passed correctness but regressed geomean to
  `0.5374411946079206` TFLOPS versus the accepted `0.5772885607891738`.
  Do not repeat this exact score-tile stride-24 skew without a materially
  different score/softmax dataflow or profiler evidence.
  A later probability-buffer stride-20 patch used
  `probabilities[kTile][kProbabilityStride]` with `kProbabilityStride = 20` and
  loaded `&probabilities[0][0]` for PV WMMA. It passed correctness but regressed
  geomean to `0.5257029292160739` TFLOPS. Do not repeat this exact stride-20
  probability skew; simple probability padding has repeatedly failed to improve
  the direct-accumulation MMA kernel.
  A later Q-fragment software-pipeline patch added `q_frag_next`, assigned
  `q_frag = q_frag_next`, and loaded the next Q fragment at the end of each QK
  chunk. It passed correctness but regressed geomean to
  `0.4215966561800913` TFLOPS versus the accepted
  `0.5772885607891738`. Do not repeat this exact Q-fragment preload chain.
  A later loop failed planning validation after three attempts because the
  returned decision omitted required fields (`expected_effect`, `risk`, and
  `next_command`). Retry feedback now explicitly asks for a complete decision
  object with all required fields even in no-edit diagnostic mode.
  A later warp-row detour added a compile-only WMMA skeleton with `mma.h`,
  shared BF16 Q/K buffers, and a `wmma::accumulator` fragment, but it never ran
  `mma_sync` or connected the fragment to online-softmax score/PV dataflow. It
  compiled with 48 BF16 registers, 1 barrier, and 17024 bytes shared memory, but
  produced no correctness or throughput evidence and was cleaned up. Do not
  repeat compile-only WMMA skeletons; a build-check patch must wire the new
  fragments into real dataflow and be intended for bounded scoring after
  compile.
  A later PV probability-fragment preload patch tried to add
  `probability_frag_next` around the eight 16-column PV chunks, but the generated
  diff duplicated the `probability_frag;` declaration statement and left a stale
  `wmma::store_matrix_sync(&output_acc[chunk_offset], output_frag, ...)` outside
  the chunk scope. NVCC failed with undefined `chunk_offset`/`output_frag` and
  parsing errors. The decision risk text already called this a duplicate store
  line and said it would cause NVCC compile failure, so the planner now treats
  that wording and the stray `probability_frag;` preload shape as invalid before
  patch application.
  A later Q double-buffer skeleton compiled cleanly but only loaded
  `q_frag_next` for the next query tile after the last key tile; it did not swap
  or consume that fragment in the QK MMA path. Compile diagnostics matched the
  accepted direct-accumulation kernel resource counts: 40 registers, 1 barrier,
  and 9920 bytes shared memory. Because it produced no correctness or throughput
  evidence and its own expected-effect/risk text called it a compile-only
  structural probe that must not be scored, the planner now rejects unused
  `q_frag_next`/`probability_frag_next` preload skeletons before compile.
  A later row-state register patch was rejected by `git apply --check` as a
  corrupt patch. The direction also repeated an unsafe per-thread register-state
  mapping in a new form: local `float row_max[kTile]`, `row_sum[kTile]`, and
  `old_scale[kTile]` arrays were initialized only by `threadIdx.x == 0`, while
  other threads would later read their own uninitialized local arrays. Do not
  move MMA row state into per-thread arrays unless every consuming thread owns
  and initializes its own row state or the state is explicitly shared.
  A later no-edit warp-row seed diagnostic on the fixed seq256/head_dim128 BF16
  suite passed correctness but remained slower than the accepted MMA kernel:
  geomean `0.4114026673771631` TFLOPS, noncausal `0.5412528843592248` TFLOPS
  at median 0.9919040203094482 ms, and causal `0.31270439311453807` TFLOPS at
  median 0.8584319949150085 ms. The causal max error was `0.015625`, higher
  than the MMA direct-accumulation causal error but still accepted by the
  current correctness gate. Do not repeat this no-patch warp-row diagnostic;
  scoring that workload again needs a structural warp-row patch.
  A later planner loop failed validation after three attempts because the
  decision text kept describing the patch as `must not be scored`. Retry
  feedback now explicitly says not to submit another compile-only skeleton with
  that language; a compile-check patch must be complete enough to score after a
  successful build, or the planner must choose a different no-edit diagnostic.
  A later no-edit `avo env` diagnostic reconfirmed the current environment:
  Anthropic SDK installed with API key present, PyTorch `2.11.0+cu130`, torch
  CUDA `13.0`, nvcc CUDA `13.0`, CUDA paths under the local cu13 package, and
  RTX A6000 target `compute_86/sm_86`. Do not spend loop steps on another
  generic environment-stability check unless a concrete build or CUDA
  environment failure occurs.
  A later 32-row MMA query-tile probe changed `kTile` to 32 and tried
  `wmma::fragment` shapes such as accumulator/matrix_a/matrix_b `32x16x16`.
  NVCC rejected those Ampere WMMA BF16 fragments as incomplete types. Keep this
  seed on supported `16x16x16` WMMA fragments; larger query tiles require
  multiple 16-row fragments or a different implementation strategy, not direct
  M=32 WMMA fragment instantiation.
  A later MMA warp-reduction helper patch inserted `__device__ __forceinline__`
  helper definitions between the `__global__ void mma_attention_kernel(` token
  and the existing parameter list, duplicating the kernel declaration and making
  NVCC report "invalid specifier on a parameter". Its own risk text said the
  patch would cause NVCC to fail and "Do not apply this patch". The planner now
  treats that wording as self-invalid before applying candidate diffs. CUDA
  helper functions are fine, but they must be complete declarations placed
  before the kernel declaration or after the kernel body, not spliced into a
  function signature.
  Planner reliability direction changed after repeated reactive guard
  checkpoints. CUDA edits should prefer `candidate_transform`, a single
  structured operation materialized by the orchestrator, rather than raw LLM
  unified diffs. Raw diffs remain a legacy fallback for edits that cannot be
  expressed as `replace_once`, `insert_before_once`, `insert_after_once`, or
  `set_constexpr_int`. Attempt history now records generalized failure classes
  such as raw-diff preflight, unsupported WMMA shape, CUDA syntax error, stale
  or undefined symbol, correctness failure, and throughput regression. Repeated
  classes should be promoted to hard preflight tracks or cause a transform-family
  change; do not add one-off phrase bans for every malformed attempt.
  A live loop after the transform redesign showed a planner-mode inconsistency:
  the decision text started with `No edit;` but also supplied a malformed
  `candidate_transform` path (`candidates / ...`). Runtime validation now
  rejects transform paths with whitespace, non-candidate roots, invalid suffixes,
  absolute paths, traversal, or backslashes during planning parse. It also
  rejects any `No edit;` decision that includes `candidate_transform` or a raw
  `candidate_patch`, so diagnostics cannot silently carry edit payloads.
  A follow-up live loop failed planning validation after three retries because
  the planner returned both `candidate_transform` and raw `candidate_patch`.
  Retry feedback now explicitly explains the mutually exclusive edit channels:
  structured-transform mode uses a transform object with `candidate_patch == ""`;
  legacy raw-diff mode omits `candidate_transform`; no-edit mode omits both.
  A later live loop got past the channel-mixing failure but repeated a recorded
  unpatched MMA seed score because older retry text still said only
  `candidate_patch`. Score-repeat errors and feedback now consistently say
  `candidate_transform/candidate_patch`, with `candidate_transform` preferred.
  A later live loop still used a legacy raw CUDA diff and scored a warp-row
  direct-global-V patch. It removed `v_tiles` and loaded V directly from global
  memory inside the PV accumulation loop. The score passed correctness but
  regressed geomean to `0.40530677363112055` TFLOPS versus the accepted MMA
  direct-accumulation `0.5772885607891738`. Runtime validation now rejects raw
  `candidate_patch` edits to `.cu`/`.cuh` files; CUDA kernel edits must use
  `candidate_transform`. Raw diffs remain available only for non-CUDA candidate
  files such as wrappers.
  A follow-up loop correctly avoided raw CUDA patch execution but still failed
  planning because the decision described a CUDA code change while providing no
  edit payload. The validation error now says `candidate_transform or
  candidate_patch must be provided` so retries are not steered back to the raw
  patch-only interface.
  The next reliability pass replaced the remaining long inline CUDA
  domain-sanity block with named structural preflight tracks. The hard checks
  now describe classes such as edit-channel integrity, WMMA fragment shape/type,
  async-copy granularity/API shape, tile-local shared-memory addressing, symbol
  lifecycle, complete shape graduation, and no-effect skeletons. Materialized
  `candidate_transform` patches run those preflights before compile/score, so
  transforms get the same structural scrutiny that raw diffs used to receive.
  The evolve loop now persists recurring promotable failure classes in
  `attempts/preflight_tracks.json` and feeds active hard tracks back into
  attempt history and command execution. Prompt context also distinguishes
  smoke-only seed caps from the actual target: long-sequence BF16 attention
  around seq 4096/8192/16384/32768, total_tokens 32768, num_heads 16, head_dim
  128, and both causal modes.
  Live-loop validation then showed the planner could describe an integer
  constant transform in prose while omitting the `candidate_transform` object.
  The parser now recovers explicit tiny constant transforms of the form
  "change NAME from OLD to NEW in candidates/.../*.cu" into `set_constexpr_int`.
  Planning validation failures are also classified into durable classes such as
  `planning_missing_edit_payload`, `planning_no_patch_compile`, and
  `planning_edit_channel` instead of falling into `unknown`, so recurring
  planner-interface failures can be promoted and summarized like execution
  failures.
  A later reliability pass found that different planning-validation failures
  shared a false command/edit fingerprint because the synthetic planning-failure
  decision always used the same `No edit; planner returned invalid decision`
  payload. Fingerprints now include the planning failure class and truncated
  validation detail, so supervisor signals distinguish edit-channel mistakes,
  missing edit payloads, and repeated no-patch compile diagnostics.
  CUDA shape graduation can require more than one tiny edit. `candidate_transform`
  now supports `op=batch` with a compact `steps_json` provider payload that the
  orchestrator parses into up to four tiny steps. Batch steps still use small
  structural transforms, not raw CUDA diffs. Supported steps now include generic
  `add_int_to_python_set`, so wrapper cap updates such as adding `512` to
  `SMOKE_SEQUENCES` are represented as a structured Python-set edit instead of
  another raw hunk. Parser recovery also infers generic uppercase Python set
  names from prose and `files_to_inspect`, not just the historical
  `SMOKE_SEQUENCES` case.
  Live loops after the batch interface repeatedly compiled the same seq512 MMA
  shape-graduation transform (`kMaxSeqLen=512` plus wrapper sequence cap `512`).
  The compile passed on sm86 with no spills, 40 registers, 1 barrier, and 9920
  bytes shared memory, but repeated compile-only probes did not advance the
  search. Attempt history now emits a follow-up signal after a successful
  compile-only structured transform, and a cross-step hard preflight rejects
  repeating that same transform with another compile command. The next live loop
  therefore scored the same seq512 batch instead of compiling again.
  Seq512 MMA shape-graduation score result on A6000: `seq_len=512`,
  `total_tokens=2048`, `num_heads=8`, `head_dim=128`, BF16, both causal modes,
  `trials=3`, `warmup=1`, `repeats=1`. Correctness passed for both cases.
  Noncausal max error `0.001953125`, median `1.3272960186004639 ms`,
  `3.2358774800882233` TFLOPS. Causal max error `0.0078125`, median
  `1.2746880054473877 ms`, `1.6847131524127585` TFLOPS. Geomean was
  `2.334850177270671` TFLOPS. Lineage did not accept it because the score shape
  differed from the current best comparison set under the old single-latest
  lineage gate. The lineage gate now tracks benchmark signatures separately, and
  that recorded seq512 score was replayed into lineage as commit
  `bc05dd85ccc8872308c786ba57664cfdffdc4640` with reason `candidate
  established benchmark case set`.
  The agent prompt now receives a compact lineage summary containing the latest
  accepted score plus all best benchmark-signature lanes, so creating a new lane
  no longer hides the earlier seq256/head_dim128 best from subsequent planning.
  The same pass loosened the unchanged-source guard only for new benchmark
  signatures; same-signature no-patch reruns remain rejected as timing-noise
  probes.
  After seeing both the seq512 and seq256 lanes, the planner proposed a seq1024
  shape-graduation batch (`kMaxSeqLen=1024`, add `1024` to `SMOKE_SEQUENCES`).
  Compile-check passed on sm86 with no spills, 40 registers, 1 barrier, and
  9920 bytes shared memory. A follow-up planning failure temporarily hid the
  compile follow-up signal, so attempt history now keeps the "score the compiled
  transform" signal alive until that transform is scored.
  The subsequent seq1024 score tried `seq_lens=1024,2048,4096` with
  `total_tokens=32768`, `num_heads=16`, `head_dim=128`, BF16, both causal modes.
  The 1024 cases passed correctness: noncausal max error `0.001953125`, median
  `27.58780860900879 ms`, `9.963745610959426` TFLOPS; causal max error
  `0.0078125`, median `27.701984405517578 ms`, `4.961339644846` TFLOPS. The
  2048 and 4096 cases failed wrapper validation because the structured transform
  only added the 1024 cap. Runtime planning now rejects MMA score commands whose
  requested `seq_lens` exceed the `kMaxSeqLen` and wrapper sequence set expressed
  by the same structured transform.
  Exa research refresh on agent failure handling reinforced the same design:
  classify failure types before deciding retry/fallback behavior, persist failure
  signatures to avoid infinite implement-verify loops, and enforce deterministic
  guardrails around probabilistic planner output. Keep future reliability work
  centered on classifier-driven preflights and search-loop routing, not on
  accumulating phrase-specific bans.
  Follow-up hardening made that promotion path operational instead of advisory.
  Active promoted failure classes are now passed into materialized transform
  preflight. For repeated `stale_or_undefined_symbol` failures, the promoted
  track rejects CUDA edits that remove a declaration while still adding uses of
  the old identifier, or that introduce duplicate local declarations in one
  edit. These are structural symbol-lifecycle checks, not exact text bans.
  WMMA fragment-shape preflight now parses added `wmma::fragment<...>` templates
  and resolves added `constexpr int` dimensions, rejecting any resolvable
  Ampere BF16 fragment dimension outside the supported 16x16x16 shape. This
  replaces narrower checks for individual historical k32/m32 spellings.
  Sequence-cap graduation is also a structural track: a patch that increases
  `kMaxSeqLen` must include the new value in `SMOKE_SEQUENCES` in the same
  wrapper/kernel edit, and wrapper-only additions beyond the accepted base cap
  are rejected before compile.
  The parser still recovers tiny integer transforms from prose, but MMA
  sequence-cap transforms are no longer allowed to be kernel-only compile
  probes. A `kMaxSeqLen` graduation outside the accepted base requires the
  wrapper sequence set in the same batch. Repeating a successful compile-only
  transform remains blocked even after that transform has been scored; the
  search must score the compiled transform or choose a different family.
  At that checkpoint, runtime source carried the accepted MMA seq1024 lane (`kMaxSeqLen=1024`
  and `SMOKE_SEQUENCES={16, 32, 64, 128, 256, 1024}`). No-edit MMA scoring below
  seq1024 is rejected so the planner does not keep spending steps on tiny smoke
  workloads. The accepted seq1024 lineage lane is commit
  `a9f55492223c977ea4525347145fe991f5e06174`: seq_len 1024, total_tokens 8192,
  num_heads 8, head_dim 128, BF16, both causal modes, geomean
  `5.073625790914057` TFLOPS.
  A source-consistency fix then found that the seq1024 source patch left the
  Python wrapper advertising `256` while the CUDA `TORCH_CHECK` only accepted
  `16/32/64/128/kMaxSeqLen`. The kernel guard now accepts any multiple of
  `kTile` up to `kMaxSeqLen`, matching the wrapper's explicit sequence set and
  future intermediate shape-graduation batches. A direct seq256 regression score
  after the fix passed correctness for both causal modes: geomean
  `0.525593999281922` TFLOPS, noncausal `0.8659655248879056` TFLOPS, causal
  `0.3190069860078135` TFLOPS.
  After that guard repair, a bounded evolve loop graduated the MMA source to the
  next shape lane with a two-step structured batch (`kMaxSeqLen=2048`, add
  `2048` to `SMOKE_SEQUENCES`). Compile passed with no spills, 40 registers,
  1 barrier, and 9920 bytes shared memory. The follow-up score at seq_len 2048,
  total_tokens 16384, num_heads 16, head_dim 128, BF16, both causal modes
  passed correctness and was accepted into lineage as commit
  `39af063`: geomean `7.079133390217197` TFLOPS, noncausal
  `9.955973678368473` TFLOPS, causal `5.033573930129197` TFLOPS. At that
  checkpoint, runtime source carried the accepted seq2048 cap.
  The first follow-up loop toward seq4096 showed a reliability issue: it
  compile-checked the seq4096 structured batch successfully, then a planning
  retry tried to score "the compiled transform" while omitting the transform
  object. Attempt-history follow-up signals now include the exact compact
  `candidate_transform` JSON, and validation feedback tells the planner to reuse
  that exact object for compiled-transform scores. With that fix, the next
  bounded loop scored and accepted the seq4096 lane in one step. Nested lineage
  commit `7162462` passed correctness at seq_len 4096, total_tokens 32768,
  num_heads 16, head_dim 128, BF16, both causal modes, with three timing trials:
  geomean `7.527287085824243` TFLOPS, noncausal `10.59316564199843` TFLOPS,
  and causal `5.348736419996861` TFLOPS. Runtime source now carries
  `kMaxSeqLen=4096` and wrapper `SMOKE_SEQUENCES` includes 4096.
  A subsequent bounded loop graduated to seq8192 with the same two-step cap
  transform (`kMaxSeqLen=8192`, add `8192` to `SMOKE_SEQUENCES`). Compile again
  passed with no spills, 40 registers, 1 barrier, and 9920 bytes shared memory.
  The follow-up score was accepted as nested lineage commit `b6622ab`: seq_len
  8192, total_tokens 32768, num_heads 16, head_dim 128, BF16, both causal
  modes, three timing trials, geomean `7.367549615485977` TFLOPS, noncausal
  `10.405911846727314` TFLOPS, causal `5.216341262175792` TFLOPS. Runtime
  source now carries `kMaxSeqLen=8192` and wrapper `SMOKE_SEQUENCES` includes
  8192. This is correctness and lane-establishment progress, not yet evidence
  of beating FA2.
- The next bounded loop graduated to seq16384 with the same two-step cap
  transform (`kMaxSeqLen=16384`, add `16384` to `SMOKE_SEQUENCES`). Compile
  passed with no spills, 40 registers, 1 barrier, and 9920 bytes shared memory.
  The follow-up score was accepted as nested lineage commit `47ddfcf`: seq_len
  16384, total_tokens 32768, num_heads 16, head_dim 128, BF16, both causal
  modes, three timing trials, geomean `7.14948482274555` TFLOPS, noncausal
  `10.08547382076117` TFLOPS, causal `5.068193536474939` TFLOPS. Runtime
  source now carries `kMaxSeqLen=16384` and wrapper `SMOKE_SEQUENCES` includes
  16384. This establishes the seq16384 target lane but is still not evidence of
  beating FA2.
- The next bounded loop graduated to seq32768 with the same two-step cap
  transform (`kMaxSeqLen=32768`, add `32768` to `SMOKE_SEQUENCES`). Compile
  passed with no spills, 40 registers, 1 barrier, and 9920 bytes shared memory.
  The follow-up score was accepted as nested lineage commit `195ea8c`: seq_len
  32768, total_tokens 32768, num_heads 16, head_dim 128, BF16, both causal
  modes, three timing trials, geomean `7.056217313717174` TFLOPS, noncausal
  `10.066887668437108` TFLOPS, causal `4.94593805139101` TFLOPS. Runtime
  source now carries `kMaxSeqLen=32768` and wrapper `SMOKE_SEQUENCES` includes
  32768. This completes initial target-shape lane coverage but is still not
  evidence of beating FA2.
- Attempt-memory promotion should operate on recurring failure classes across the
  current unaccepted tail, not only exact back-to-back repeats. The loop now
  counts classified failures since the last accepted result, persists every
  promotable class that reaches the repeat threshold in `preflight_tracks.json`,
  and reloads those classes before materialized CUDA transform/patch preflight.
  This moves recurring CUDA mistakes toward hard structural tracks without
  adding another exact historical phrase ban.
- The next reliability pass made the hard-preflight mechanism more structural
  and less prompt-specific. The planner context now treats small seed caps as
  safety fences rather than optimization targets, and retry feedback no longer
  injects exact historical "do not repeat this phrase" wording for recorded
  failures. Promoted state now records the concrete checks activated by each
  recurring class; recurring CUDA syntax failures enable delimiter-completeness
  preflight, recurring WMMA shape failures enable explicit fragment-shape
  contract preflight, and recurring symbol-lifecycle failures enable removed
  and duplicate declaration checks. MMA shape-contract changes also have an
  always-on wrapper/kernel batch preflight so wrapper-only or kernel-only
  graduation attempts fail before compile.
