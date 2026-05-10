# CUDA Kernel Design Practice

This file gives the AVO planner broad CUDA programming practice. It is not an
Ampere attention recipe and should not be treated as local benchmark evidence.
Use it to form better hypotheses before proposing a kernel transform.

## The Basic Mental Model

- A GPU is a throughput machine. It usually wins when the problem exposes many
  independent operations and enough memory traffic or math to keep many SMs
  busy.
- CUDA code is heterogeneous: host code allocates memory, moves data, launches
  kernels, and synchronizes; device code runs as many lightweight GPU threads.
- Kernel launch overhead and host/device transfers matter. For small problems,
  a mathematically faster kernel can lose to launch overhead, synchronization, or
  transfer cost.
- Good CUDA design normally starts by asking what work item each thread, warp,
  and block owns. A constant change is not a work-mapping change unless it also
  changes the indexing, data ownership, synchronization, or resource usage.
- Latency hiding comes from resident warps and independent instructions. A
  kernel that has too little parallelism, too many dependencies, or too many
  resource constraints can leave issue slots idle even if each thread does
  useful work.

## Decomposing Work

- Start from the output shape. Decide whether one thread, one warp, one block,
  or multiple blocks should own each output element, row, tile, or reduction.
- Prefer ownership patterns that minimize cross-thread communication. If a
  result can be computed by one warp with warp shuffles, that may be simpler
  than a block-wide reduction. If a result needs a whole tile, make the block
  own the tile and keep the producer/consumer relationship local.
- Blocks must be independently executable in any order. A transform that needs
  one block to wait for another block inside the same normal kernel is usually
  wrong unless it is rewritten around a valid cooperative launch or atomic
  protocol.
- Use enough blocks to fill the GPU. A kernel with only a few blocks may be
  correct and fast per block but still slow overall because most SMs are idle.
- Keep block sizes aligned with warps. Non-multiples of 32 are legal, but the
  final partial warp wastes lanes for the whole block lifetime.
- Map `threadIdx.x` to the fastest-moving dimension when that dimension should
  be contiguous across lanes. CUDA linearizes block indices with `x` fastest,
  then `y`, then `z`, which affects warp lane assignment and memory coalescing.

## Indexing And Data Layout

- Separate logical coordinates from physical addresses. A good transform should
  state both: which output tile or data item is owned, and how that owner maps to
  memory.
- Bounds predicates are part of the algorithm. Tail handling must preserve
  correctness for non-divisible shapes and must avoid out-of-bounds reads even
  when the out-of-range value is later masked.
- Coalescing is a warp property. Consecutive lanes should generally access
  consecutive, naturally aligned data when reading or writing global memory.
- Data layout can matter more than arithmetic. Array-of-structs can be awkward
  when each warp needs one field from many elements; struct-of-arrays can be
  better for coalesced field loads. The right choice depends on the access
  pattern, not naming style.
- Vectorized loads and stores are useful only when the addresses are aligned,
  contiguous, and guarded at the tail. A vectorized out-of-bounds load is still a
  bug even if the extra values are ignored.
- Avoid integer index recomputation in hot loops when simple loop structure or
  pointer increments can preserve clarity and reduce instruction count.

## Memory Movement

- Global memory is high bandwidth but high latency. The goal is not simply to
  reduce the number of load instructions; it is to reduce wasted transactions,
  improve reuse, and keep enough independent work to hide latency.
- Shared memory is a block-local scratchpad, not an automatic cache. It helps
  when it creates reuse, changes a bad access pattern into a coalesced one,
  performs a layout transform, or stages data for tensor-core or warp-level
  operations.
- A shared-memory tile has a full dataflow: cooperative global load, optional
  layout conversion or padding, synchronization, reuse by consumers, and a
  second synchronization before the storage is overwritten if consumers may
  still read it.
- Shared memory can be slower than using registers or caches when it adds copies
  and barriers without enough reuse. A shared tile that is written once and read
  once by the same thread is usually not a useful tile.
- Shared-memory banks matter. If many lanes hit the same bank, accesses
  serialize. Padding one dimension, changing layout, or swizzling addresses can
  turn a bank-conflicted tile into a useful tile.
- Registers are the fastest storage for per-thread values, but more live values
  increase register pressure. Spills to local memory usually mean device-memory
  traffic hidden behind a thread-local name.
- Constant memory is for small read-only values that are broadly reused. It is
  not a replacement for large tensor operands.

## Synchronization And Communication

- Use `__syncthreads()` when data written by one thread or warp in a block may be
  read by another. The barrier must be reached by all live threads in the block.
- Use warp-level synchronization only for warp-local cooperation. A warp-level
  primitive is not enough when another warp can produce or consume the data.
- Avoid divergent barriers. A condition around `__syncthreads()` must be
  block-uniform, or the kernel can deadlock or produce undefined behavior.
- Warp shuffles are often a good fit for reductions or broadcasts inside one
  warp because they avoid shared memory and block-wide barriers.
- Atomics solve update races, not necessarily algorithm design. They can
  serialize, change numerical order, and require explicit memory-order thinking
  when used for inter-block protocols.
- Cross-block reductions are usually split into multiple kernels or use a
  carefully designed library primitive. A single ordinary kernel has no global
  barrier.

## Tiling Pattern

- Tiling is a semantic transformation, not just an array declaration. A tile
  should define ownership, load mapping, storage layout, synchronization, compute
  reuse, and store mapping.
- A simple tiled loop often has this shape: load tile from global to shared,
  synchronize, compute on the tile, synchronize if shared storage will be
  reused, advance to the next tile.
- Double buffering adds another invariant: one stage is consumed while a
  different stage is filled. The code must make stage identity, wait points, and
  overwrite safety obvious.
- `cp.async` or `cuda::memcpy_async` should be tied to a producer/consumer
  pipeline. An immediate copy, commit, wait, and consume sequence often preserves
  correctness but does not create meaningful overlap.
- Async-copy transforms must handle alignment, access size, tail predicates, and
  zero-fill or guarded shared state. The pipeline is only useful if consumers
  never read uninitialized or stale stage data.
- Prefer vectorized async-copy groups for throughput, but do not turn copy
  granularity alone into a hard rejection. Let coherent async-copy dataflow
  changes reach compile/repair unless they violate a structural invariant such
  as stage lifecycle, address ownership, or initialized-data guarantees.
- Tiling can increase arithmetic intensity, but it also spends shared memory,
  registers, instructions, and barriers. Measure the tradeoff.

## Tensor Cores And Matrix Operations

- Tensor cores are fed by warp-level matrix instructions. A tensor-core transform
  changes how warps load fragments, issue matrix multiply-accumulate operations,
  and store or reduce accumulators.
- WMMA fragments are opaque. Code should not assume a stable per-lane register
  layout inside a fragment. Treat fragments as values manipulated through the
  supported load, MMA, fill, and store APIs or through architecture-specific MMA
  instructions with a known contract.
- Fragment shape, operand layout, data type, and leading dimension are coupled.
  Changing one field without changing the matching load/store and MMA contract
  is not a coherent tensor-core transform.
- Tensor-core kernels normally need data layout work. The hard part is often
  arranging global/shared memory so the warp can feed matrix instructions
  efficiently while preserving coalescing and avoiding bank conflicts.
- For production-grade matrix and attention kernels, library designs such as
  CUTLASS are useful because they encode tiling, copy pipelines, MMA shapes,
  shared-memory layouts, and epilogues as one coherent design instead of many
  isolated edits.

## Launch Configuration And Resource Tradeoffs

- Occupancy is constrained by registers per thread, shared memory per block,
  threads per block, blocks per SM, and architectural warp limits.
- Maximum occupancy is not automatically maximum performance. More occupancy can
  hide latency, but lower occupancy with better locality, fewer instructions, or
  more tensor-core utilization can win.
- Low occupancy is still a warning when the profiler shows latency stalls or too
  few eligible warps. The fix might be lower register pressure, less shared
  memory, smaller blocks, more independent work, or a different work mapping.
- Read compiler output. Register count, spill stores/loads, shared-memory size,
  barriers, and stack/local memory often explain why a source-level transform did
  not behave as expected.
- Launch configuration should match the work shape. Changing block size without
  changing indexing and per-thread responsibility often just changes resource
  pressure and lane utilization.

## Streams, Events, And Host/Device Overlap

- CUDA API calls and kernel launches can be asynchronous with respect to the
  host. This is useful only when the program has independent CPU work, memory
  copies, or kernels to overlap.
- A stream is an ordered queue of work. Operations in one stream execute in
  order; work in different streams may overlap when dependencies and device
  resources allow it.
- The legacy default stream can implicitly synchronize with other blocking
  streams. Unintended default-stream work can erase expected overlap.
- CUDA events are a normal way to time GPU work and express dependencies between
  streams. Host wall-clock timing without proper synchronization can measure
  enqueue time instead of execution time.
- Host/device transfer optimization is separate from kernel optimization.
  Moving less data, batching transfers, using pinned memory for asynchronous
  copies, and keeping data resident on the GPU can matter as much as a faster
  kernel.

## Profiling And Optimization Workflow

- Use realistic workloads. Optimizing a tiny shape can select for launch
  overhead, cache effects, or toy control flow that is irrelevant to target
  shapes.
- Establish correctness first, then baseline performance, then profiler
  evidence. A faster wrong kernel is not progress, and a regression with clear
  measurements is useful search data.
- Classify the bottleneck before editing: memory bandwidth, memory transactions,
  shared-memory bank conflicts, register pressure, low occupancy, dependency
  stalls, synchronization, branch divergence, instruction mix, or tensor-core
  under-utilization.
- Start profiling with broad sections such as launch, occupancy, high-level
  compute/memory utilization, scheduler behavior, and memory workload. Drill
  into source counters, instruction mix, or memory tables when the broad view
  identifies a plausible limiter.
- Compare distributions, not one timing sample. Use warmups, repeated samples,
  stable inputs, same device, same clocks when possible, and the same benchmark
  settings across candidates.
- Negative results should update the hypothesis. If a correct transform is
  slower, record whether it likely added barriers, increased register pressure,
  reduced coalescing, caused bank conflicts, worsened occupancy, or missed the
  actual bottleneck.

## Semantic Transform Guidance For The Planner

- Make the smallest coherent transformation that preserves invariants and can be
  validated. "Small" means scoped, reviewable, recoverable, and tied to a clear
  hypothesis; it does not mean the smallest textual edit.
- A good CUDA transform changes one semantic axis: work mapping, tile shape,
  operand layout, staging path, synchronization boundary, accumulator placement,
  vectorization contract, async pipeline depth, or launch-resource balance.
- A weak transform changes a constant or inserts scaffolding while leaving the
  actual work ownership, memory traffic, and dataflow unchanged.
- Every transform should state expected benefit and risk. Examples: "coalesce
  this strided load by staging a transposed tile"; "reduce barriers by making
  sharing warp-local"; "increase arithmetic intensity by reusing K/V tile across
  multiple Q rows"; "risk: higher shared memory may reduce occupancy."
- Preserve invariants explicitly: address bounds, alignment, initialized values,
  synchronization, mask semantics, accumulation precision/order, output layout,
  supported fragment shapes, and launch feasibility.
- When a transform fails, classify the failure by the violated invariant or
  bottleneck hypothesis. Promote recurring classes into structural preflight
  checks only when the class is general enough to prevent a family of bad edits.

## Useful Planner Queries

- Retrieval query: `CUDA work decomposition thread block tile warp row mapping coalesced layout`.
- Retrieval query: `CUDA indexing threadIdx x fastest linearization data layout tail predicates`.
- Retrieval query: `CUDA memory movement shared tile global load sync reuse bank conflict padding`.
- Retrieval query: `CUDA synchronization __syncthreads warp shuffle atomics cross block reduction`.
- Retrieval query: `CUDA tiling double buffering cp.async memcpy_async pipeline producer consumer overlap`.
- Retrieval query: `CUDA tensor core WMMA fragments opaque shape layout leading dimension`.
- Retrieval query: `CUDA launch configuration occupancy registers shared memory block size ptxas`.
- Retrieval query: `CUDA streams events asynchronous host device default stream synchronization`.
- Retrieval query: `CUDA profiling realistic workloads SpeedOfLight occupancy memory workload roofline`.
- Retrieval query: `CUDA semantic transform smallest coherent transformation invariants hypothesis`.

## Sources

- NVIDIA CUDA Programming Guide, programming model:
  https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
- NVIDIA CUDA Programming Guide, writing CUDA SIMT kernels:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- NVIDIA CUDA Programming Guide, asynchronous execution:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
- NVIDIA CUDA Programming Guide, pipelines:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html
- NVIDIA CUDA Programming Guide, asynchronous data copies:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
- NVIDIA CUDA C++ Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
