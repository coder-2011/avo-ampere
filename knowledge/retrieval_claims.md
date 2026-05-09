# Retrieved Knowledge Claims

This file defines the high-value information currently in the local AVO knowledge
corpus. Each claim is useful only if it can steer a future transform decision,
reject a bad transform family, or explain a score/gate result.

## Ampere Target And Baseline

- Claim: the target is NVIDIA RTX A6000 / sm86, so the kernel should use Ampere
  primitives (`cp.async`, `mma.sync`, warp reductions, BF16 tensor cores), not
  Blackwell-only TMA, WGMMA, or FA4 pipelines.
- Evidence source: local architecture notes, NVIDIA Ampere material, and the
  runtime env/compile target.
- Why useful: prevents the planner from proposing architecturally impossible
  Blackwell edits.
- Retrieval query: `Ampere sm86 A6000 cp.async mma.sync no TMA WGMMA FA4`.

- Claim: FlashAttention-2 is the comparison baseline, but not the lineage
  acceptance threshold.
- Evidence source: local architecture notes and `knowledge/ampere.md`.
- Why useful: allows incremental candidate improvements while still tracking
  the gap to FA2.
- Retrieval query: `FlashAttention-2 baseline comparison lineage acceptance threshold`.

## FA2/CUTLASS Directional Cues

- Claim: CUTLASS's Ampere FlashAttention v2 example uses 128x128 M/N tiles, 128
  threads, 16-byte contiguous alignment, Q/K/V shared-memory staging through
  `cp.async`, swizzled shared-memory layouts, Ampere tensor-core MMA, and
  integrated online softmax.
- Evidence source: Exa research over NVIDIA CUTLASS's Ampere FlashAttention v2
  example and local `knowledge/ampere.md` notes.
- Why useful: defines a productive structural direction for the local seed:
  move toward FA2-like tile/copy/layout/dataflow structure rather than constant
  retunes.
- Retrieval query: `CUTLASS Ampere FlashAttention v2 128x128 128 threads swizzled online softmax`.

- Claim: FA2/SM80 code derives `NumThreads` from tiled MMA structure; a
  block-size constant is not a proxy for implementing a new workload
  distribution.
- Evidence source: Exa research over Dao-AILab FlashAttention SM80 code and
  local `knowledge/ampere.md` notes.
- Why useful: blocks misleading one-line thread-count edits that claim new
  query-tile or split-work dataflow.
- Retrieval query: `FlashAttention SM80 NumThreads tiled MMA workload distribution`.

- Claim: FA2's SM80 forward kernel couples Q staging to tiled shared-memory
  layouts, optional Q-in-register copies, and K/V `cp.async` copy/fence/wait
  phases; a standalone synchronous `q_shared` allocation is not equivalent to
  the FA2 dataflow.
- Evidence source: Exa research over Dao-AILab FlashAttention SM80 forward
  kernel and local Q-staging regression.
- Why useful: redirects future staging attempts toward Q-in-register reuse,
  K/V async pipeline structure, or broader Q/K/V layout changes instead of
  isolated synchronous Q shared-memory staging.
- Retrieval query: `FlashAttention SM80 Q in regs Share_Q_K_smem cp_async K V pipeline q_shared regression`.

## General CUDA Grounding

- Claim: CUDA kernel design starts from the execution hierarchy: grids contain
  blocks, blocks contain threads, and hardware executes threads in 32-lane warps
  under the SIMT model.
- Evidence source: NVIDIA CUDA Programming Guide and `knowledge/b/cuda_general.md`.
- Why useful: helps the planner reason about whether a proposed work mapping
  changes real parallel execution or just changes a constant.
- Retrieval query: `CUDA execution model grid block thread warp SIMT divergence`.

- Claim: CUDA memory spaces have different scopes and costs: global memory is
  grid-visible and persistent, shared memory is block-local scratchpad storage,
  registers are thread-local, and local memory is thread-local in scope but
  device-memory backed.
- Evidence source: NVIDIA CUDA Programming Guide and `knowledge/b/cuda_general.md`.
- Why useful: prevents confusing "local" memory with fast storage and helps the
  planner choose between registers, shared memory, and global memory.
- Retrieval query: `CUDA memory spaces global shared register local constant cache`.

- Claim: global-memory performance depends on warp-level coalescing; consecutive
  lanes reading consecutive words use transactions efficiently, while strided or
  scattered lanes waste bandwidth.
- Evidence source: NVIDIA CUDA Programming Guide, NVIDIA CUDA Best Practices
  Guide, and `knowledge/b/cuda_general.md`.
- Why useful: gives the planner a general reason to inspect address patterns
  before proposing a shared-memory or vectorized-copy change.
- Retrieval query: `CUDA global memory coalescing warp consecutive lanes transactions`.

- Claim: shared memory is useful when it creates reuse, coalesces otherwise poor
  global access, performs a needed layout transform, or feeds a warp/tensor-core
  operation; it must include a correct producer/consumer synchronization story.
- Evidence source: NVIDIA CUDA Programming Guide, NVIDIA CUDA Best Practices
  Guide, and `knowledge/b/cuda_general.md`.
- Why useful: stops the planner from treating shared-memory staging as free and
  connects staging proposals to a concrete benefit.
- Retrieval query: `CUDA shared memory synchronization bank conflicts tiling`.

- Claim: a new shared-memory staging buffer is not a semantic CUDA transform by
  itself; the same materialized transform must also load, store, or consume that
  buffer in executable dataflow.
- Evidence source: rejected loop after general CUDA grounding and runtime
  `no_effect_shared_staging_buffer` preflight.
- Why useful: rejects no-effect staging scaffolds while preserving real
  shared-memory staging transforms.
- Retrieval query: `CUDA shared staging buffer no effect must be loaded stored consumed`.

- Claim: occupancy is a resource tradeoff constrained by block size, registers,
  shared memory, and hardware resident-block/warp limits; maximum occupancy is
  not automatically maximum performance.
- Evidence source: NVIDIA CUDA Programming Guide, NVIDIA Ampere tuning guide,
  and `knowledge/b/cuda_general.md`.
- Why useful: discourages blind thread-count or register-cap retunes and asks for
  a bottleneck hypothesis plus measurements.
- Retrieval query: `CUDA occupancy registers shared memory spills ptxas`.

- Claim: CUDA optimization should be measurement driven: establish correctness,
  time with warmups/replicates, profile bottlenecks, make one coherent transform,
  and treat regressions as search evidence.
- Evidence source: NVIDIA CUDA Best Practices Guide, NVIDIA Nsight Compute
  profiling guide, local AVO loop design, and `knowledge/b/cuda_general.md`.
- Why useful: aligns future planner steps with how CUDA programmers actually
  improve kernels rather than accumulating reactive phrase bans.
- Retrieval query: `CUDA optimization workflow hypothesis measure profile transform`.

- Claim: CUDA work decomposition starts from output ownership and the mapping of
  work to threads, warps, blocks, and tiles; a constant edit is not a real
  work-mapping change unless indexing, data ownership, synchronization, or
  resources change with it.
- Evidence source: NVIDIA CUDA Programming Guide and
  `knowledge/b/cuda_programming_practice.md`.
- Why useful: keeps planner proposals focused on semantic work ownership rather
  than tiny textual edits that do not change the kernel's execution structure.
- Retrieval query: `CUDA work decomposition thread block tile warp row mapping coalesced layout`.

- Claim: CUDA indexing and layout decisions are coupled: `threadIdx.x` is the
  fastest-linearized block coordinate, warp-lane memory addresses determine
  coalescing, and tail predicates must protect non-divisible shapes from
  out-of-bounds accesses.
- Evidence source: NVIDIA CUDA Programming Guide and
  `knowledge/b/cuda_programming_practice.md`.
- Why useful: gives the planner a general way to reason about address patterns
  before changing vectorization, tile shapes, or work mapping.
- Retrieval query: `CUDA indexing threadIdx x fastest linearization data layout tail predicates`.

- Claim: a useful shared-memory tile has a complete dataflow: cooperative global
  load, optional layout conversion or padding, synchronization, compute reuse,
  and overwrite safety; otherwise shared memory can just add copies and
  barriers.
- Evidence source: NVIDIA CUDA C++ Best Practices Guide and
  `knowledge/b/cuda_programming_practice.md`.
- Why useful: encourages semantic staging transforms and rejects scaffolding
  that declares storage without changing memory traffic or reuse.
- Retrieval query: `CUDA memory movement shared tile global load sync reuse bank conflict padding`.

- Claim: CUDA synchronization choices should match the communication scope:
  block-shared data needs block-wide barriers, warp-local reductions can often
  use shuffles, atomics do not replace algorithm design, and ordinary kernels do
  not have a global barrier.
- Evidence source: NVIDIA CUDA Programming Guide and
  `knowledge/b/cuda_programming_practice.md`.
- Why useful: helps future transforms preserve correctness while reducing
  unnecessary synchronization or avoiding invalid cross-block dependencies.
- Retrieval query: `CUDA synchronization __syncthreads warp shuffle atomics cross block reduction`.

- Claim: `cp.async`/`cuda::memcpy_async` is most useful as part of a
  producer/consumer pipeline with double buffering or equivalent overlap; an
  immediate copy/commit/wait sequence often preserves correctness without adding
  meaningful latency hiding.
- Evidence source: NVIDIA CUDA Programming Guide pipelines/asynchronous-copy
  chapters and `knowledge/b/cuda_programming_practice.md`.
- Why useful: steers async-copy attempts toward real overlap and complete stage
  invariants instead of cosmetic instruction substitution.
- Retrieval query: `CUDA tiling double buffering cp.async memcpy_async pipeline producer consumer overlap`.

- Claim: tensor-core transforms must treat WMMA fragments as opaque and keep
  fragment shape, operand layout, data type, leading dimension, loads, MMA
  operations, and stores as one coherent contract.
- Evidence source: NVIDIA CUDA Programming Guide WMMA/Tensor Core material,
  local WMMA preflight failures, and `knowledge/b/cuda_programming_practice.md`.
- Why useful: prevents isolated fragment-shape or layout edits that compile
  poorly or break tensor-core dataflow.
- Retrieval query: `CUDA tensor core WMMA fragments opaque shape layout leading dimension`.

- Claim: streams and events express host/device and inter-stream ordering:
  operations in one stream are ordered, work in separate streams may overlap,
  legacy default-stream behavior can add implicit synchronization, and GPU
  timings need event or synchronization discipline.
- Evidence source: NVIDIA CUDA Programming Guide asynchronous-execution chapter
  and `knowledge/b/cuda_programming_practice.md`.
- Why useful: lets the planner distinguish kernel-internal changes from
  host-side launch, timing, and overlap issues.
- Retrieval query: `CUDA streams events asynchronous host device default stream synchronization`.

- Claim: CUDA profiling should use realistic workloads and broad bottleneck
  classification first: launch configuration, occupancy, SpeedOfLight
  compute/memory utilization, scheduler behavior, memory workload, source
  counters, and timing distributions.
- Evidence source: NVIDIA CUDA C++ Best Practices Guide, NVIDIA Nsight Compute
  Profiling Guide, and `knowledge/b/cuda_programming_practice.md`.
- Why useful: discourages optimizing tiny unrealistic shapes and ties future
  transforms to measured bottleneck classes.
- Retrieval query: `CUDA profiling realistic workloads SpeedOfLight occupancy memory workload roofline`.

- Claim: small CUDA edits should be small coherent semantic transformations:
  scoped, reviewable, recoverable, tied to a hypothesis, and preserving concrete
  invariants such as bounds, alignment, synchronization, masks, accumulation,
  supported fragment shapes, and launch feasibility.
- Evidence source: local design critique, CUDA programming practice notes, and
  `knowledge/b/cuda_programming_practice.md`.
- Why useful: captures the desired planner behavior directly: avoid raw diffs
  and tiny no-effect text edits while still allowing meaningful kernel evolution.
- Retrieval query: `CUDA semantic transform smallest coherent transformation invariants hypothesis`.

## Transform Interface Lessons

- Claim: `set_constexpr_int` and Python set updates are contract-only
  transforms. They are valid for real constant or shape-contract retunes, but
  they do not implement new dataflow, tiling, staging, scheduling, split-Q work,
  or async pipelines.
- Evidence source: failed planning attempts and runtime semantic-alignment
  preflight.
- Why useful: aligns the planner with "small coherent semantic move" instead of
  "tiny text move".
- Retrieval query: `semantic transform mismatch contract-only set_constexpr_int dataflow staging scheduling`.

- Claim: recurring failure classes should be classified and promoted to
  structural preflights instead of adding phrase-specific bans.
- Evidence source: attempt history and current runtime preflight tracks.
- Why useful: keeps reliability work general and search-loop oriented.
- Retrieval query: `recurring failure class promote hard preflight structural track`.

- Claim: a structurally rejected `candidate_transform` invalidates any older
  compile-only "score this transform" follow-up for the same transform identity.
- Evidence source: loop after the shared-staging-buffer preflight and runtime
  attempt-history follow-up logic.
- Why useful: prevents the planner from repeatedly trying to score a stale
  no-effect transform after preflight already proved it is invalid.
- Retrieval query: `candidate_transform structural preflight rejection invalidates pending compile-only score followup`.

- Claim: ambiguous `replace_once` and `insert_*_once` transform anchors should be
  repaired by using larger unique anchors with surrounding code; runtime now
  reports matching start line numbers to make that repair actionable.
- Evidence source: K-staging attempts after the rejected-followup fix and
  runtime transform materialization errors.
- Why useful: helps the planner repair semantic transforms without falling back
  to raw CUDA diffs.
- Retrieval query: `candidate_transform ambiguous anchor matching start lines larger unique anchor`.

## CUDA Structural Constraints

- Claim: future `cp.async` attempts must use aligned 16-byte groups, treat 16
  bytes as 8 BF16 elements, keep scalar tails disjoint, preserve zero-fill or
  guarded shared state for partial tiles, and add real overlap rather than an
  immediate copy/commit/wait sequence.
- Evidence source: NVIDIA CUDA/CUTLASS references plus failed local async-copy
  attempts.
- Why useful: turns several recurring malformed async-copy attempts into a
  constructive structural contract.
- Retrieval query: `Ampere cp.async 16-byte aligned groups scalar BF16 async copy real dataflow`.

- Claim: Ampere BF16 WMMA fragment edits should keep the supported 16x16x16
  contract in this WMMA seed unless the kernel is deliberately rewritten around
  a different TensorOp interface.
- Evidence source: failed local WMMA fragment attempts and runtime
  `wmma_fragment_shape` preflight.
- Why useful: prevents unsupported fragment shapes and incomplete-type compile
  failures.
- Retrieval query: `wmma_fragment_shape Ampere BF16 fragment dimension outside supported 16x16x16`.

## Search Evidence

- Claim: the current best accepted local candidate is the MMA seed with
  `kThreads=128`, full-target correctness, and geomean `7.777584666360881`
  TFLOPS.
- Evidence source: accepted lineage and experiment log.
- Why useful: anchors the gate and makes later regressions interpretable.
- Retrieval query: `best accepted kThreads 128 geomean 7.777584666360881 full target`.

- Claim: `kThreads=64` is correct but slower, scoring geomean
  `7.587127963961811` TFLOPS versus best `7.777584666360881`.
- Evidence source: rejected loop after semantic-alignment fix.
- Why useful: weakens the "fewer idle threads improves occupancy" hypothesis for
  this seed.
- Retrieval query: `kThreads 64 rejected geomean 7.587127963961811 occupancy`.

- Claim: isolated synchronous Q shared-memory staging is correct but slower,
  scoring geomean `6.722112165053056` TFLOPS versus best
  `7.777584666360881`.
- Evidence source: rejected Q-staging loop.
- Why useful: future staging should be more FA2-like: 16-byte copy granularity,
  swizzled/shared layouts matching MMA access, and overlap or broader K/V/Q
  dataflow instead of isolated synchronous Q-tile loading.
- Retrieval query: `synchronous Q shared memory staging regression geomean 6.722112165053056`.

- Claim: the repaired full-target `q_shared` Q-staging transform compiled and
  passed correctness, but regressed geomean to `6.686302249012325` TFLOPS while
  increasing shared memory to 14016 bytes.
- Evidence source: loop after transform-anchor repair and compiled-transform
  scoring.
- Why useful: confirms that isolated synchronous Q staging is not the current
  winning direction; future shared-memory work should combine Q/K/V layout,
  vectorized copies, swizzling, or async overlap rather than only moving Q loads
  into shared memory.
- Retrieval query: `q_shared Q staging repaired anchors geomean 6.686302249012325 shared memory 14016`.

- Claim: WMMA chunk-loop unroll-by-2 preserved full-target correctness but
  regressed geomean to `7.758592599549404` TFLOPS versus best
  `7.777584666360881`.
- Evidence source: loop after rejected-followup clearing and chunk-unroll score.
- Why useful: discourages more local unroll-only edits and points the search
  back toward dataflow, layout, tiling, and staging changes.
- Retrieval query: `WMMA chunk unroll by 2 rejected geomean 7.758592599549404`.
