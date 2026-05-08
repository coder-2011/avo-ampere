# AVO Ampere

AVO Ampere is the executable research scaffold for evolving Ampere-targeted attention kernels. It is not a finished faster-than-FlashAttention result. The repository currently contains the reliability layer needed before autonomous kernel mutation is safe: hardware checks, isolated scoring, candidate loading, lineage gates, bounded agent commands, and CUDA extension smoke tests.

This repo is paired with [`coder-2011/avo`](https://github.com/coder-2011/avo), which holds the paper, architecture notes, and experiment log. `avo-ampere` is the runtime implementation track.

## Current status

- Target hardware: NVIDIA RTX A6000 / Ampere, compute capability `sm_86`.
- Target workload: BF16 forward attention with head dimension 128 and sequence lengths 4096, 8192, 16384, and 32768.
- Baseline: FlashAttention-2. FlashAttention-4 is intentionally excluded because its Blackwell path uses primitives that are not available on Ampere.
- Candidate support: Python candidate modules plus a first CUDA-extension smoke candidate.
- Agent support: Anthropic-backed variation planning with strict schema validation and a bounded command allowlist.
- Research state: infrastructure-first checkpoint. The code can score and gate candidates, but the repository does not yet contain a novel accepted attention kernel.

## What was built

Recent commits show the work moved in layers:

- `feat: scaffold ampere AVO runtime` created the package, CLI, config model, isolated execution, transcript handling, lineage repository flow, and Ampere knowledge notes.
- `chore: checkpoint fa2 baseline seed env verification` added environment and baseline verification paths for A6000 work.
- `feat: enforce strict agent decisions` constrained the agent output to a validated decision schema.
- `feat: add bounded variation executor` added `run-decision`, which executes only selected `avo` subcommands without a shell.
- `feat: add candidate scoring backend` added the candidate interface and a PyTorch SDPA seed candidate.
- `fix: harden Anthropic agent planning` improved structured-tool fallbacks and validation.
- `feat: add CUDA extension candidate smoke` added a minimal compiled CUDA extension path that copies an SDPA result, proving the candidate build/load path before replacing the attention computation itself.

## Repository layout

```text
avo/                         Python package and CLI
  agent.py                   Anthropic variation-decision contract
  benchmark.py               Torch, FlashAttention, and candidate scoring
  cli.py                     User-facing commands
  compile.py                 CUDA compilation helper
  config.py                  Ampere attention-case configuration
  evolve.py                  Bounded decision executor
  isolation.py               Child-process score isolation
  lineage.py                 Candidate score gating and git-backed lineage
  transcript.py              Transcript compaction helpers
candidates/
  torch_sdpa_seed.py         Correctness seed that delegates to PyTorch SDPA
  cuda_identity_seed.py      CUDA-extension smoke candidate
  cuda_identity/             Minimal PyTorch/CUDA extension source
kernels/smoke.cu             NVCC smoke source
tests/                       Unit coverage for the orchestration layer
knowledge/ampere.md          Ampere-specific constraints and assumptions
```

## Candidate interface

Candidate scoring loads a Python module from `--candidate` and calls:

```python
attention(q, k, v, causal: bool)
```

Inputs and outputs use PyTorch SDPA layout:

```text
(batch, heads, sequence, head_dim)
```

The candidate must return a tensor matching PyTorch SDPA output for the same inputs. Crashes and import failures are converted into failed score records by the isolated worker instead of crashing the orchestrator.

## Common commands

Install development dependencies with `uv`, then run the checks relevant to the machine:

```bash
uv run --extra dev pytest
uv run python -m avo compile --source kernels/smoke.cu --out-dir /tmp/avo-build
uv run --extra cuda python -m avo env
```

Score the PyTorch seed backend:

```bash
uv run --extra cuda python -m avo score \
  --backend torch-sdpa \
  --seq-lens 4096 \
  --causal both \
  --repeats 3 \
  --warmup 1
```

Score a Python candidate:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/torch_sdpa_seed.py \
  --seq-lens 4096 \
  --causal both \
  --repeats 3 \
  --warmup 1
```

Score the CUDA-extension smoke candidate:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_identity_seed.py \
  --seq-lens 4096 \
  --causal false \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 300
```

Score the naive CUDA attention-math smoke candidate on a tiny shape:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_naive_attention_seed.py \
  --seq-lens 16 \
  --total-tokens 16 \
  --num-heads 1 \
  --head-dim 16 \
  --dtype fp32 \
  --causal both \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 300
```

Seed a FlashAttention-2 baseline lineage:

```bash
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=2 \
  uv run --extra baseline python -m pip install flash-attn --no-build-isolation

uv run --extra cuda --extra baseline python -m avo seed-baseline ./lineage \
  --backend flash-attn \
  --seq-lens 4096,8192,16384,32768 \
  --repeats 3 \
  --warmup 1
```

## Agent workflow

The agent wrapper uses the Anthropic API and expects `ANTHROPIC_API_KEY` in the environment.

```bash
uv run python -m avo agent-plan \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md
```

Persist the returned JSON decision, then run the bounded command from it:

```bash
uv run python -m avo run-decision decision.json --attempt-json attempts/latest.json
uv run python -m avo evolve-once --lineage ./lineage --knowledge knowledge/ampere.md --step-json attempts/step.json
```

`run-decision` intentionally accepts only selected `avo env`, `avo compile`, and `avo score` commands. It does not run arbitrary shell, git, file-editing, or destructive commands.

`evolve-once` runs one validated agent decision, records the step, and commits only score payloads that pass the existing lineage gate.
Agent prompts include a concise local repo context so decisions prefer existing candidate files over upstream-only paths.

## What is still missing

- A real CUDA attention candidate that replaces SDPA rather than wrapping it.
- Scaling the naive attention candidate beyond tiny correctness smokes.
- A complete mutation loop that edits candidate code safely.
- Performance evidence beating FlashAttention-2 on the target A6000 cases.
- Longer lineage history with accepted and rejected candidates.

The important progress in this repo is the safety and measurement substrate. The kernel-search result is still open.
