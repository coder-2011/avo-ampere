# AVO Ampere

AVO Ampere is the executable research scaffold for evolving Ampere-targeted attention kernels. It is not a finished faster-than-FlashAttention result. The repository contains the reliability layer needed before autonomous kernel mutation is safe: hardware checks, isolated scoring, candidate loading, lineage gates, bounded agent commands, structured transform materialization, and CUDA extension candidates.

This repo is paired with [`coder-2011/avo`](https://github.com/coder-2011/avo), which holds the paper, architecture notes, and experiment log. `avo-ampere` is the runtime implementation track.

## Current status

- Target hardware: NVIDIA RTX A6000 / Ampere, compute capability `sm_86`.
- Target workload: BF16 forward attention with head dimension 128 and sequence lengths 4096, 8192, 16384, and 32768.
- Baseline: FlashAttention-2. FlashAttention-4 is intentionally excluded because its Blackwell path uses primitives that are not available on Ampere.
- Candidate support: Python candidate modules plus CUDA-extension attention candidates, including a BF16 WMMA QK/PV seed accepted through the seq32768 lane.
- Agent support: Anthropic-backed variation planning with strict schema validation, a bounded command allowlist, a local searchable knowledge-corpus retriever, structured candidate transforms for CUDA edits, and a candidate-only patch application substrate.
- Scoring support: optional replicate timing via `--trials`; per-case TFLOPS uses the median timed sample and records timing noise, benchmark settings, target, and environment metadata in JSON.
- Attempt memory: `evolve-once --attempts-dir` and `evolve-loop --attempts-dir` record accepted and rejected steps outside the committed lineage, classify failure classes, and persist recurring classes as active hard preflight tracks in `preflight_tracks.json`, including the concrete structural checks activated by each promoted class.
- Research state: the autonomous loop has accepted benchmark lanes across the full target shape set through seq32768. The open result is still optimizing that seed toward beating FlashAttention-2 on the target suite.

## What was built

Recent commits show the work moved in layers:

The project now has a working Ampere-focused AVO runtime for evolving CUDA attention kernels. It includes a Python package, CLI, typed configuration, isolated candidate execution,
transcript capture, lineage tracking, and a small CUDA knowledge base for guiding the search loop.

The system can verify the local A6000 environment, run a known FA2-style baseline, and score candidate implementations through a controlled candidate interface. Early candidates include
a PyTorch SDPA seed, a compiled CUDA extension smoke path, and a BF16 WMMA attention seed that gives the search loop a real tensor-core starting point instead of beginning from arbitrary
CUDA text.

The agent loop is now constrained around structured decisions rather than free-form shell or raw patch behavior. It validates plans, executes only bounded avo subcommands, materializes
structured candidate transforms, and records classified failures so recurring CUDA issues can become preflight checks instead of one-off prompt warnings.

The CUDA side has also gained structural guardrails for common failure classes: malformed transforms, invalid WMMA fragment shapes, syntax errors, duplicate declarations, symbol-
lifecycle mistakes, and wrapper/kernel sequence-cap mismatches. The MMA path has been advanced through bounded loops to accepted large-sequence coverage, including seq2048 through
seq32768, while keeping the search process reviewable and recoverable.

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
knowledge/b/cuda_general.md  General CUDA execution, memory, profiling, and optimization grounding
knowledge/b/cuda_programming_practice.md
                             General CUDA kernel design and optimization practice
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

Search the local knowledge corpus that is supplied to the planner:

```bash
uv run python -m avo knowledge-search knowledge/ampere.md \
  --query "Ampere cp.async shared staging WMMA head_dim 128"
```

`agent-plan`, `evolve-once`, and `evolve-loop` use the same deterministic lexical
retriever internally. The query is built from the current lineage summary, recent
attempt history, and local repo context, so the planner receives relevant snippets
from `knowledge/` instead of the entire corpus as one flat prompt block.
`knowledge/retrieval_claims.md` defines the high-value claims in that corpus,
including their evidence source, why each claim is useful, and the query expected
to retrieve it. General CUDA grounding lives under `knowledge/b/`, while
Ampere/attention-specific search evidence lives in `knowledge/ampere.md`.

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

Score the BF16 WMMA QK/PV attention candidate on the accepted seq32768 lane:

```bash
uv run --extra cuda python -m avo score \
  --backend candidate \
  --candidate candidates/cuda_mma_attention_seed.py \
  --seq-lens 32768 \
  --total-tokens 32768 \
  --num-heads 16 \
  --head-dim 128 \
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

Run a bounded profiler diagnostic on a candidate when profiler evidence would
change the next transform choice:

```bash
uv run --extra cuda python -m avo profile \
  --backend candidate \
  --candidate candidates/cuda_mma_attention_seed.py \
  --seq-lens 4096,8192,16384,32768 \
  --total-tokens 32768 \
  --num-heads 16 \
  --head-dim 128 \
  --dtype bf16 \
  --causal both \
  --launch-count 1 \
  --timeout-s 300
```

`avo profile` wraps the same isolated score worker in Nsight Compute. It is a
diagnostic command, not an acceptance path: it prints structured JSON with
profiler settings, score payload when available, and a profiler error such as
`ncu_not_found`, `profiler_permission`, `profiler_unsupported_runtime`, or
`timeout`. In this Thunder-backed runtime, Nsight/CUPTI profiling is reported as
unavailable before launch so the evolve loop does not hang on unsupported
profiling.

Seed a FlashAttention-2 baseline lineage:

```bash
uv run --extra cuda python -m avo env

eval "$(uv run --extra cuda python -m avo baseline-env)"
uv pip install flash-attn --no-build-isolation

uv run --extra cuda python -m avo seed-baseline ./lineage \
  --backend flash-attn \
  --seq-lens 4096,8192,16384,32768 \
  --total-tokens 32768 \
  --num-heads 16 \
  --head-dim 128 \
  --dtype bf16 \
  --causal both \
  --trials 3 \
  --repeats 1 \
  --warmup 1 \
  --timeout-s 900 \
  --force
```

Check the `baseline_build` block from `avo env` before installing FA2. Source builds need
`ok_for_torch_extension_build: true`; a major `torch_cuda`/`nvcc_cuda` mismatch will fail inside
PyTorch extension setup before any attention benchmark can run. When a matching Python-installed
NVIDIA CUDA root is present, AVO's baseline worker selects it even if the host image exports an
incompatible system `CUDA_HOME`. The `baseline` extra intentionally does not install `flash-attn`
because package resolution would build it before AVO can apply `FLASH_ATTN_CUDA_ARCHS=80`; use the
explicit `uv pip install` command above after `baseline-env` has exported the CUDA root, compile
limits, library paths, and the local `libcudart.so` link shim. The `cuda` extra pins the CUDA 13
nvcc, CRT, NVVM, and CCCL header wheels needed for Torch `cu130` extension builds; the candidate
wrapper uses the same CUDA-root and link-shim logic for extension builds.

The FA2 baseline remains a comparison lane, not the candidate acceptance threshold. Candidate
lineage commits are accepted against prior candidate scores with the same benchmark signature so
the search can preserve incremental progress before it reaches FA2 throughput. Lineage summaries
include a derived `baseline_comparisons` section whenever a candidate lane and FlashAttention-2
baseline lane share the same benchmark signature; this records the candidate/FA2 ratio and the
remaining TFLOPS gap without changing the acceptance gate.

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

`run-decision` intentionally accepts only selected `avo env`, `avo compile`,
`avo profile`, and `avo score` commands. It does not run arbitrary shell, git,
file-editing, or destructive commands.

Candidate source edits have a separate manual substrate:

```bash
uv run python -m avo apply-patch candidate.patch --dry-run
uv run python -m avo apply-patch candidate.patch
```

`apply-patch` reads a raw unified diff, extracts `diff --git` paths, rejects paths outside `candidates/`, rejects path traversal, symlink-mode patches, binary patches, renames, deletes, and mode changes, then runs `git apply --check --whitespace=error` before applying. It does not stage, commit, score, or bypass the lineage gate.

Anthropic decisions now choose one edit mode. `no_edit` runs only bounded
diagnostics. `transform` carries a structured `candidate_transform` such as
`replace_once`, `replace_block_once`, `insert_before_once`,
`insert_after_once`, `set_constexpr_int`, or a coherent `batch` with a native
`steps` array; this is the preferred path for CUDA kernel evolution.
`replace_block_once` is for coherent loop/body/helper replacement when the
semantic move is larger than a one-line expression swap. `steps_json` remains a
legacy fallback for older records and plain-JSON responses, but the Anthropic
tool schema exposes structured batch steps directly.
`legacy_patch` remains available for raw candidate diffs but raw `.cu`/`.cuh`
kernel edits are rejected there so kernel evolution stays reviewable and
recoverable through structured semantic moves. When `evolve-once` applies an
edit but the step is not accepted by the score gate, it checks and applies the
reverse patch so rejected edits do not pollute the next attempt. When a
candidate step is accepted, the lineage commit records `sources/latest/...`
snapshots for the scored candidate module and companion source directory
alongside `scores/latest.json`; direct local Python imports under `candidates/`
and Python modules dynamically imported from the same `candidates/` tree, plus
statically declared or runtime-observed
`torch.utils.cpp_extension.load(sources=[...])` files are included in that
snapshot. Candidate modules can also expose
`AVO_SOURCE_FILES` or `__avo_source_files__` as a path or iterable of paths for
runtime-discovered CUDA/source files, such as dynamically assembled extension
sources. `sources/latest/manifest.json` records path, size, and hash metadata
for audit. Accepted edited steps also record
`patches/latest.patch`.

`evolve-once` runs one validated agent decision, records the step, and commits only score payloads that pass the suite-aware lineage gate. Candidates compare against the best prior score with the same benchmark case signature; a new signature can establish its own lane when correctness passes and source artifacts are captured.
Agent prompts include a concise local repo context so decisions prefer existing candidate files over upstream-only paths.
Planner prompts also have a final character budget over dynamic sections: bulky
repo context, knowledge, lineage, and attempt history are compacted before the
Anthropic request, with attempt history tail-preserved so repair requests and
pending `candidate_transform` JSON survive long runs. If the planner returns an
invalid decision, the validation-feedback retry uses the same budget and
preserves the prompt head, newest context tail, and validation error.
If a candidate edit fails transform materialization, compile, or score correctness, the evolve step can ask the agent for an immediate revised executable edit after reverting the failed patch.
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

`evolve-loop` requires `--attempts-dir` so cross-step memory is always available. It stops when a step is accepted, when rejected-patch cleanup fails, when a planner provider/API outage is recorded, or when `--max-steps` is exhausted. Command failures and gate rejections are recorded, summarized into the next prompt, and allowed to continue until one of those stop conditions is reached. Provider outages stop the loop after the recorded step because repeating them cannot improve CUDA search. Attempt summaries also append a supervisor signal when the recent history shows repeated unaccepted command/edit fingerprints, recurring failure classes in the unaccepted tail, or five unaccepted attempts in a row; recurring promotable classes are written to `preflight_tracks.json` and loaded before materialized transform/patch preflight.

When an applied edit fails compile, transform materialization, or correctness, the loop can make a bounded immediate repair request before finalizing the step. The failed edit is reverted first, the repair prompt includes the current compiler/correctness/materialization error plus earlier failed repair payloads from that episode, and an unchanged replay of any failed episode payload is rejected before execution.

## What is still missing

- Optimizing the accepted seq32768 WMMA QK/PV seed beyond shape coverage toward FA2-competitive throughput.
- Scaling the warp-row attention seed beyond tiny correctness smokes.
- Longer-running autonomous supervision beyond the capped `evolve-loop`, including richer active intervention beyond the current attempt-memory signals, promoted checks, and provider-outage stop policy.
- Automatic capture of generated source files that are never imported, passed through `torch.utils.cpp_extension.load`, or reported through the explicit source-file manifest hook.
- Performance evidence beating FlashAttention-2 on the target A6000 cases.
- Longer lineage history with accepted candidates and a larger rejected-attempt search trajectory.

The important progress in this repo is the safety and measurement substrate. The kernel-search result is still open.
