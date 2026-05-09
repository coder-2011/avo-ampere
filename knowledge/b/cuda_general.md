# General CUDA Working Knowledge

This file is general CUDA grounding for the AVO planner. It should help the
agent reason like a CUDA programmer before it reasons about Ampere attention
details. Prefer these notes for broad kernel-design decisions; use
`cuda_programming_practice.md` for broader CUDA optimization workflow and use
`knowledge/ampere.md` for architecture-specific constraints and local search
evidence.

## Execution Model

- A CUDA kernel is launched by the host as a grid of thread blocks. Blocks are
  scheduled onto streaming multiprocessors (SMs), and multiple blocks can reside
  on one SM when registers, shared memory, resident-block limits, and resident
  warp limits allow it.
- A block is the normal unit of cooperation. Threads in the same block can share
  block-local shared memory and synchronize with `__syncthreads()`. Ordinary
  CUDA blocks cannot globally synchronize with other blocks during the same
  kernel launch; cross-block phase boundaries usually require separate kernel
  launches, atomics with carefully designed protocols, or cooperative launch
  features.
- Threads are grouped into warps of 32 lanes. Consecutive linear thread IDs are
  assigned to the same warp. A block size that is not a multiple of 32 leaves a
  partially populated warp and usually wastes issue slots.
- CUDA uses SIMT execution: each thread has its own state and control flow, but
  warp lanes issue instructions together. Divergent branches inside a warp
  serialize paths and reduce useful lane utilization. Predication or warp-uniform
  branch conditions are often better than arbitrary per-lane control flow.
- The common indexing pattern is to map `blockIdx`, `blockDim`, and `threadIdx`
  to a global element or tile coordinate, then guard out-of-range threads. Bounds
  checks are part of correctness, not optional cleanup.

## Memory Spaces

- Global memory is large, persistent across kernel launches, visible to all
  threads, and relatively high latency. Kernels usually read inputs from global
  memory and write final results back to global memory.
- Shared memory is a programmer-managed block-local scratchpad located on the
  SM. It is much smaller than global memory and has lower latency/higher
  bandwidth, but it only lives for one block during one kernel launch.
- Registers are private to a thread and are the preferred place for live scalar
  state. Too many live values increase register pressure and can reduce
  occupancy.
- Local memory is thread-local in scope but physically backed by device memory.
  It often appears when arrays or spills cannot stay in registers. Treat local
  memory usage as a performance warning unless it is deliberate and measured.
- Constant memory is read-only from kernels and useful for small values reused
  broadly. It is not a substitute for large per-element data.
- L1/cache behavior and shared-memory capacity share hardware resources on many
  GPUs, so adding shared-memory staging can reduce effective cache capacity.
  Shared memory is useful when it creates reuse, coalesces otherwise poor global
  accesses, supports layout transforms, or enables a real pipeline.

## Global Memory Access

- Global memory is served in memory transactions for a warp. Consecutive lanes
  reading consecutive words use bandwidth efficiently; large strides or scattered
  addresses force extra transactions and waste bandwidth.
- Coalescing is about the addresses requested by a warp, not just about each
  scalar load looking valid. A technically correct kernel can be slow if each
  lane fetches from a distant address.
- Vectorized and naturally aligned loads/stores can reduce instruction count and
  help memory throughput, but alignment and tail handling must be explicit.
- Avoid redundant global reads when the data can be reused from registers,
  shared memory, or warp shuffles. Do not add shared memory just because data is
  read from global memory; add it when reuse or layout conversion pays for the
  extra copy and synchronization.

## Shared Memory And Synchronization

- Shared memory is a cooperation mechanism. Every producer/consumer relationship
  through shared memory needs a clear synchronization story.
- Use `__syncthreads()` when data written by one warp or thread may be read by
  another warp or thread in the same block. Use warp-level synchronization only
  when the sharing is truly warp-local.
- A conditional `__syncthreads()` is only safe when all live threads in the block
  execute the same barrier. Divergent barriers can deadlock or create undefined
  behavior.
- Shared memory is banked. Access patterns that map many lanes to the same bank
  serialize and can erase the benefit of staging. Padding, swizzled layouts, or
  different tile shapes are common fixes.
- Shared-memory tiling is most useful when it converts uncoalesced global access
  into coalesced loads/stores, eliminates repeated global reads, or feeds
  tensor-core/warp-level operations in the layout they require.

## Occupancy And Resources

- Occupancy is the number of resident warps/blocks relative to hardware limits.
  It is constrained by block size, registers per thread, static shared memory,
  dynamic shared memory, and architectural resident-block/warp limits.
- Higher occupancy can hide latency, especially for memory-bound kernels, but
  maximum occupancy is not automatically maximum performance. More occupancy can
  reduce per-thread registers or shared memory and make each warp less efficient.
- Register pressure is a first-class design constraint. A small code change that
  increases live ranges can reduce resident warps or create spills to local
  memory.
- Read ptxas output after compile. Registers, shared memory, spills, barriers,
  and stack/local memory often explain performance before timing does.
- Use occupancy calculators or APIs to understand launch feasibility and rough
  residency, then validate with profiling and timing. Occupancy estimates do not
  replace measurement.

## Profiling And Measurement

- Start with a correct reference implementation and a small benchmark that
  catches numerical errors. Speed measurements are meaningless until correctness
  is stable.
- Time kernels with CUDA events or an equivalent GPU-side timing path, include
  warmups, use repeated samples, and compare medians or distributions rather than
  one-off timings.
- Keep comparisons on the same device, dtype, input shape, launch settings,
  warmup/repeat/trial settings, and clock/driver context when possible.
- Use profiling to classify the bottleneck before editing: memory bandwidth,
  memory transactions, shared-memory bank conflicts, occupancy/resource limits,
  warp divergence, instruction mix, synchronization stalls, or tensor-core
  utilization.
- Nsight Compute is most useful when scoped to a small number of representative
  kernels. Start with high-level utilization/Speed-of-Light style sections, then
  drill into memory workload, scheduler, source counters, occupancy, and
  instruction stats as needed.

## How CUDA Programmers Usually Approach Optimization

- Work from a hypothesis. "This change should improve X because measurement Y
  says X is limiting" is useful; "try this optimization because it is common" is
  weak.
- Change one coherent behavior at a time: a tile shape, a memory layout, a copy
  path, a synchronization boundary, an accumulator placement, or a warp/block
  work mapping. The edit should be reviewable and reversible.
- Preserve invariants while moving performance-sensitive code: index bounds,
  memory alignment, producer/consumer synchronization, initialized shared state,
  mask semantics, numerical accumulation order, and output layout.
- Prefer transformations that expose more parallelism, reduce wasted memory
  traffic, improve locality, improve tensor-core feeding, or remove unnecessary
  synchronization.
- Treat negative results as data. A correct regression can show that the
  bottleneck hypothesis was wrong, that synchronization/copy overhead dominates,
  or that the transform needs a broader dataflow change.
- Do not infer that a smaller constant, fewer threads, more shared memory, more
  unrolling, or a new async copy is good by itself. Each is a resource tradeoff.
- For CUDA attention kernels, good changes usually preserve online-softmax
  correctness while improving Q/K/V tiling, global-to-shared copy structure,
  shared-memory layout, tensor-core MMA feeding, accumulator placement, or
  overlap between memory movement and compute.

## Useful Planner Queries

- Retrieval query: `CUDA execution model grid block thread warp SIMT divergence`.
- Retrieval query: `CUDA memory spaces global shared register local constant cache`.
- Retrieval query: `CUDA global memory coalescing warp consecutive lanes transactions`.
- Retrieval query: `CUDA shared memory synchronization bank conflicts tiling`.
- Retrieval query: `CUDA occupancy registers shared memory spills ptxas`.
- Retrieval query: `CUDA profiling Nsight Compute bottleneck memory occupancy scheduler`.
- Retrieval query: `CUDA optimization workflow hypothesis measure profile transform`.

## Sources

- NVIDIA CUDA Programming Guide, writing CUDA SIMT kernels:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- NVIDIA CUDA C++ Programming Guide, hardware multithreading and performance
  guidance:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- NVIDIA CUDA C++ Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA Nsight Compute Kernel Profiling Guide:
  https://docs.nvidia.com/nsight-compute/2023.1/ProfilingGuide/
