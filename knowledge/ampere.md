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

## Baseline

- Seed and baseline should be FlashAttention-2.
- Correctness reference should be PyTorch `scaled_dot_product_attention`.
- Benchmark forward pass with BF16, head dimension 128, sequence lengths
  4096/8192/16384/32768, total tokens 32768, causal and non-causal.
- Baseline installation should pin FlashAttention compile targets for A6000 via
  `FLASH_ATTN_CUDA_ARCHS=80` (the upstream build script’s Ampere-family target)
  and cap build parallelism with `MAX_JOBS`.

## Search Space

- `cp.async` pipeline depth and staging.
- Shared-memory layouts and bank-conflict reduction.
- Register pressure and spill avoidance.
- Warp-level online softmax reductions.
- Instruction scheduling around `mma.sync` latency.
- Split-Q versus split-K work partitioning.

## Gate

A candidate enters lineage only when it passes correctness and matches or
improves the current best aggregate score. Failed attempts remain in the agent
run log, not the committed lineage.
