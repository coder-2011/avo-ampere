# AVO Ampere

AVO Ampere is the executable research scaffold for evolving Ampere-targeted attention kernels. It is not a finished faster-than-FlashAttention result. The repository currently contains the reliability layer needed before autonomous kernel mutation is safe: hardware checks, isolated scoring, candidate loading, lineage gates, bounded agent commands, and CUDA extension smoke tests.

This repo is paired with [`coder-2011/avo`](https://github.com/coder-2011/avo), which holds the paper, architecture notes, and experiment log. `avo-ampere` is the runtime implementation track.

## Current status

- Target hardware: NVIDIA RTX A6000 / Ampere, compute capability `sm_86`.
- Target workload: BF16 forward attention with head dimension 128 and sequence lengths 4096, 8192, 16384, and 32768.
- Baseline: FlashAttention-2. FlashAttention-4 is intentionally excluded because its Blackwell path uses primitives that are not available on Ampere.
- Candidate support: Python candidate modules plus a first CUDA-extension smoke candidate.
- Agent support: Anthropic-backed variation planning with strict schema validation, a bounded command allowlist, and a candidate-only patch application substrate.
- Scoring support: optional replicate timing via `--trials`; per-case TFLOPS uses the median timed sample and records timing noise, benchmark settings, target, and environment metadata in JSON.
- Attempt memory: `evolve-once --attempts-dir` and `evolve-loop --attempts-dir` record accepted and rejected steps outside the committed lineage and feed recent summaries back into later agent prompts.
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
- `feat: add tiny mma attention seed` added a 16/32-token BF16 WMMA QK/PV candidate so the local search has a tensor-core attention-math foothold.

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
  cuda_mma_attention_seed.py BF16 WMMA QK/PV attention smoke candidate
  cuda_identity/             Minimal PyTorch/CUDA extension source
  cuda_mma_attention/        Tiny tensor-core attention source
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
uv run --extra agent --extra cuda python -m avo env --env-file ../avo/.env.local
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

Score the tiny tiled online-softmax smoke candidate on a tiny shape:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_tiled_attention_seed.py \
  --seq-lens 16 \
  --total-tokens 16 \
  --num-heads 1 \
  --head-dim 16 \
  --dtype bf16 \
  --causal both \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 300
```

Score the tiny warp-row online-softmax smoke candidate on a tiny shape:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_warp_rows_attention_seed.py \
  --seq-lens 16 \
  --total-tokens 16 \
  --num-heads 1 \
  --head-dim 16 \
  --dtype bf16 \
  --causal both \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 300
```

Score the tiny BF16 WMMA QK/PV attention smoke candidate on its two-tile 32-token shape:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_mma_attention_seed.py \
  --seq-lens 32 \
  --total-tokens 32 \
  --num-heads 1 \
  --head-dim 16 \
  --dtype bf16 \
  --causal both \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 300
```

For noisier comparisons, add `--trials 5` or higher. Each case record will include
the raw timing samples, min, median, mean, and coefficient of variation; the
reported `milliseconds` and `tflops` use the median sample. Each score summary
also includes a `benchmark` block with warmup/repeat/trial settings, seed,
A6000/sm86 target metadata, Python/PyTorch/CUDA versions, and visible GPU
properties so accepted lineage commits can be audited later.

Seed a FlashAttention-2 baseline lineage:

```bash
uv run --extra cuda python -m avo env

FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=1 NVCC_THREADS=1 \
  uv pip install flash-attn --no-build-isolation

uv run --extra cuda --extra baseline python -m avo seed-baseline ./lineage \
  --backend flash-attn \
  --seq-lens 4096,8192,16384,32768 \
  --repeats 3 \
  --warmup 1
```

Check the `baseline_build` block from `avo env` before installing FA2. Source builds need
`ok_for_torch_extension_build: true`; a major `torch_cuda`/`nvcc_cuda` mismatch will fail inside
PyTorch extension setup before any attention benchmark can run.

## Agent workflow

The agent wrapper uses the Anthropic API and expects `ANTHROPIC_API_KEY` in the environment.
Use `avo env --env-file PATH` to check the Anthropic SDK import and key presence without printing
the key value.

```bash
uv run --extra agent python -m avo env --env-file ../avo/.env.local
uv run python -m avo agent-plan \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --attempts-dir ./attempts
```

Persist the returned JSON decision, then run the bounded command from it:

```bash
uv run python -m avo run-decision decision.json --attempt-json attempts/latest.json
uv run python -m avo evolve-once \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --attempts-dir ./attempts \
  --step-json attempts/latest-step.json
```

`run-decision` intentionally accepts only selected `avo env`, `avo compile`, and `avo score` commands. It does not run arbitrary shell, git, file-editing, or destructive commands.

Candidate source edits have a separate manual substrate:

```bash
uv run python -m avo apply-patch candidate.patch --dry-run
uv run python -m avo apply-patch candidate.patch
```

`apply-patch` reads a raw unified diff, extracts `diff --git` paths, rejects paths outside `candidates/`, rejects path traversal, symlink-mode patches, binary patches, renames, deletes, and mode changes, then runs `git apply --check --whitespace=error` before applying. It does not stage, commit, score, or bypass the lineage gate.

Anthropic decisions now include a required `candidate_patch` string. Empty means no edit. A non-empty raw unified diff is applied through the same validator before `run-decision` or `evolve-once` runs the bounded `next_command`; the command allowlist remains limited to `avo env`, `avo compile`, and `avo score`. When `evolve-once` applies a patch but the step is not accepted by the score gate, it checks and applies the reverse patch so rejected edits do not pollute the next attempt. When a candidate step is accepted, the lineage commit records `sources/latest/...` snapshots for the scored candidate module and companion source directory alongside `scores/latest.json`; patched accepted steps also record `patches/latest.patch`.

`evolve-once` runs one validated agent decision, records the step, and commits only score payloads that pass the existing lineage gate.
Agent prompts include a concise local repo context so decisions prefer existing candidate files over upstream-only paths.
When `--attempts-dir` is provided, `evolve-once` also writes a timestamped step JSON for every run, including rejected and failed attempts. Later `agent-plan` or `evolve-once` calls summarize the latest records from that directory so the agent can avoid repeating known dead ends without adding them to committed lineage.

Run a bounded multi-step session by repeating the same safe one-step unit:

```bash
uv run python -m avo evolve-loop \
  --lineage ./lineage \
  --knowledge knowledge/ampere.md \
  --attempts-dir ./attempts \
  --max-steps 3 \
  --loop-json attempts/latest-loop.json
```

`evolve-loop` requires `--attempts-dir` so cross-step memory is always available. It stops when a step is accepted, when rejected-patch cleanup fails, or when `--max-steps` is exhausted. Command failures and gate rejections are recorded, summarized into the next prompt, and allowed to continue until one of those stop conditions is reached. Attempt summaries also append a supervisor signal when the recent history shows repeated unaccepted command/edit fingerprints or five unaccepted attempts in a row; the signal asks the agent to reset strategy but does not bypass the bounded command or lineage gate.

## What is still missing

- Scaling the tiny WMMA QK/PV seed beyond its 16/32-token smoke shapes.
- Scaling the warp-row attention seed beyond tiny correctness smokes.
- Longer-running autonomous supervision beyond the minimal capped `evolve-loop`, including active intervention and richer stop policies.
- Broader dependency manifests for accepted candidates that import files outside the scored module and companion source directory.
- Performance evidence beating FlashAttention-2 on the target A6000 cases.
- Longer lineage history with accepted candidates and a larger rejected-attempt search trajectory.

The important progress in this repo is the safety and measurement substrate. The kernel-search result is still open.
