# AVO Ampere

Minimal Agentic Variation Operator scaffold for attention-kernel evolution on
NVIDIA RTX A6000 / Ampere (`sm_86`).

This repository is intentionally separate from the paper/reference repository.
It starts with the reliability substrate: Ampere-aware configuration, isolated
score execution, PyTorch SDPA correctness checks, FlashAttention-2 baseline hooks,
lineage gating, transcript compaction, and an Anthropic variation-agent contract.

## Target

- Hardware: NVIDIA RTX A6000, compute capability 8.6.
- Compile target: `sm_86`.
- Seed and comparison baseline: FlashAttention-2.
- Excluded baseline: FlashAttention-4, because its Blackwell path depends on
  primitives not available on Ampere.
- Benchmark family: BF16 forward attention, head dimension 128, total tokens
  32768, sequence lengths 4096/8192/16384/32768, causal and non-causal.

## Baseline and Lineage

- `init-lineage <path>` initializes a standalone git repo used for candidate history.
- `seed-baseline <path> --backend flash-attn ...` runs FlashAttention-2 in an
  isolated worker, writes `scores/baseline.json`, and initializes `scores/latest.json`.
- `commit-score <lineage-path> <score-json>` applies the gate against latest accepted
  score and commits only if correctness and geomean are non-regressing.

Recommended FA2 install target for A6000 in this scaffold:

```bash
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=2 uv run --extra baseline python -m pip install flash-attn --no-build-isolation
```

## Quick Checks

```bash
uv run --extra dev pytest
uv run python -m avo compile --source kernels/smoke.cu --out-dir /tmp/avo-build
uv run --extra cuda python -m avo env
uv run --extra cuda python -m avo score --backend torch-sdpa --seq-lens 4096 --causal both --repeats 3 --warmup 1
uv run --extra cuda --extra baseline python -m avo seed-baseline ./lineage --backend flash-attn --seq-lens 4096,8192,16384,32768 --repeats 3 --warmup 1
```

`score` runs the CUDA work in a child Python process and parses a structured
`AVO_RESULT_JSON=...` line from the worker. A crashing worker should return a
failed score record instead of taking down the orchestrator.

## Agent Use

The variation-agent wrapper stays in the Anthropic ecosystem and reads
`ANTHROPIC_API_KEY` from the environment. For local development, source the
paper repo's `.env.local` before running agent commands.
Variation decisions prefer Anthropic strict tool use and fall back to validated
JSON text when a local SDK/model path does not support that request shape.

```bash
set -a && source /home/ubuntu/avo/.env.local && set +a
uv run python -m avo agent-plan --lineage ./lineage --knowledge knowledge/ampere.md
```

The initial agent command produces a structured plan. Persist that JSON before
execution, then run a bounded allowlisted command from it:

```bash
uv run python -m avo run-decision decision.json --attempt-json attempts/latest.json
```

`run-decision` accepts only selected `avo ...` subcommands and executes them as
`python -m avo ...` without a shell. Code-editing tools and autonomous mutation
loops should be added only after scoring and lineage gates are stable.
