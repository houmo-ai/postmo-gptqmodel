# Native EBSS Integration Plan

## Goal

Replace the current script-and-monkey-patch style MoE EBSS calibration generation flow with a repository-native implementation that:

- keeps the EBSS math unchanged in spirit
- avoids global monkey patching of transformers.GenerationMixin
- avoids subprocess-based generation as the primary path
- supports direct in-memory calibration data generation for quantization
- remains extensible across MoE model families

## Current State Summary

Current moe_ebss relies on two invasive mechanisms:

1. Model patching to expose per-step expert routing counts
2. Global monkey patching of beam search to inject EBSS penalties into search

This works, but it is hard to maintain, leaks behavior globally, and makes reuse inside the quantization pipeline awkward.

## Target Architecture

### 1. Expert Statistics Adapter Layer

Introduce a model-specific adapter interface to expose expert routing statistics in a uniform way.

Responsibilities:

- attach(model, num_beams)
- enable_collection()
- disable_collection()
- get_step_counts() -> per-layer per-beam expert counts
- clear_step_counts()
- reorder_state(beam_idx)
- detach()

Planned implementations:

- Qwen3MoeExpertStatsAdapter
- Qwen3NextExpertStatsAdapter
- future: Glm4MoeLiteExpertStatsAdapter

### 2. EBSS Search Core

Centralize the algorithm that converts expert usage statistics into beam penalties.

Inputs:

- cumulative expert counts per layer/beam/expert
- current beam scores
- logits/log_probs
- beam reorder indices

Core math:

- accumulate expert counts over decode steps
- compute variance over experts per layer
- average variance across layers
- normalize across beams
- apply penalty to beam scores

Planned extension points:

- penalty_strength (lambda)
- warmup schedule for penalty
- count-based or frequency-based variance mode

### 3. Local Beam Search Backend

Implement a repository-local EBSS beam search backend used only by calibration generation.

Requirements:

- preserve current beam search behavior as much as possible
- maintain KV cache reorder correctness
- reorder EBSS expert-count state together with beam state
- avoid patching transformers globally

This backend should be callable only from the calibration generation provider path.

### 4. Native Calibration Generation Provider

Upgrade MoeEbssProvider to use in-process generation instead of subprocess script invocation.

Primary path:

- provider generates text rows in memory
- runtime passes generated rows directly to prepare_calibration_dataset

Optional path:

- provider may also export JSONL when configured for debugging or reproducibility

### 5. Runtime Integration

Keep QuantizeConfig.calibration_generation as the entry point.

Runtime behavior:

- if disabled: no change
- if enabled with provider=moe_ebss: call native provider
- prefer in-memory generated rows
- optional JSONL export remains secondary

## Implementation Phases

## Phase 1: Isolate EBSS Math

### Task 1.1
Create a new module for EBSS scoring logic.

New file:
- gptqmodel/quantization/calibration/generation/ebss_score.py

Contents:
- cumulative expert count update helpers
- variance computation helpers
- beam penalty application helpers

### Task 1.2
Add unit tests for EBSS score math.

New tests:
- tests/test_ebss_score.py

Coverage:
- zero-count stability
- variance over experts
- layer averaging
- beam normalization
- penalty application in log-score form

## Phase 2: Introduce Expert Stats Adapters

### Task 2.1
Create adapter base interface.

New file:
- gptqmodel/quantization/calibration/generation/expert_stats_adapter.py

### Task 2.2
Implement Qwen3-MoE adapter.

New file:
- gptqmodel/quantization/calibration/generation/adapters/qwen3_moe.py

### Task 2.3
Implement Qwen3-Next adapter.

New file:
- gptqmodel/quantization/calibration/generation/adapters/qwen3_next.py

### Task 2.4
Add adapter registry.

New file:
- gptqmodel/quantization/calibration/generation/adapters/registry.py

### Task 2.5
Add adapter tests.

New tests:
- tests/test_expert_stats_adapter_registry.py
- tests/test_qwen3_next_adapter.py
- tests/test_qwen3_moe_adapter.py

## Phase 3: Localize Beam Search

### Task 3.1
Create a local EBSS beam search implementation.

New file:
- gptqmodel/quantization/calibration/generation/ebss_beam_search.py

Requirements:
- no global patching
- explicit dependency on expert stats adapter
- explicit handling of beam reorder

### Task 3.2
Add beam-search-focused tests.

New tests:
- tests/test_ebss_beam_search_state.py

Coverage:
- beam reorder keeps expert counts aligned
- no active-layer case is stable
- penalty path does not corrupt output shapes

## Phase 4: Replace Subprocess Provider Path

### Task 4.1
Refactor MoeEbssProvider to call native generator functions directly.

Files:
- gptqmodel/quantization/calibration/generation/moe_ebss_provider.py

Behavior:
- resolve model adapter
- create generation runner
- produce text rows in memory
- optionally export JSONL

### Task 4.2
Add provider integration tests.

New tests:
- tests/test_moe_ebss_provider_native.py

Coverage:
- in-memory generation path
- optional export path
- unsupported model failure path

## Phase 5: Runtime and Quantization Integration

### Task 5.1
Update runtime resolution to prefer in-memory rows.

Files:
- gptqmodel/quantization/calibration/generation/runtime.py

Behavior:
- if provider returns rows, pass rows directly
- if provider returns artifact with export path only, load JSONL as fallback

### Task 5.2
Keep example scripts compatible.

Files:
- examples/quantization/examples/example_glm_4_7_flash.py

Behavior:
- generated calibration path should not require a prebuilt local dataset
- optional export path remains configurable

### Task 5.3
Add integration tests.

New tests:
- tests/test_quantize_calibration_generation_runtime.py

Coverage:
- enabled path uses generated rows directly
- disabled path preserves old behavior
- allow_override behavior

## Phase 6: Remove Legacy Script Dependence Gradually

### Task 6.1
Retain old scripts as compatibility/debug tools only.

Files kept but demoted:
- gptqmodel/quantization/calibration/moe_ebss/qwen3_moe_gen.py
- gptqmodel/quantization/calibration/moe_ebss/qwen3_next_gen.py

### Task 6.2
Update README/docs to mark native path as primary.

Files:
- gptqmodel/quantization/calibration/README.md
- optional docs under docs/

## Acceptance Criteria

- EBSS calibration generation works without global monkey patching
- MoeEbssProvider supports direct in-memory generation
- Expert routing statistics remain correctly aligned under beam reorder
- Existing calibration_generation config remains the public entry point
- Optional JSONL export still works for debugging/reproducibility
- Existing non-generation quantization flows remain unchanged

## Test Plan

### Unit
- EBSS score math
- adapter registry and adapter-specific counting
- local beam search state transitions

### Integration
- native provider in-memory generation
- runtime generation-to-prepare_calibration_dataset flow
- quantize path with generation enabled

### Regression
- tests/qcfg/test_activation.py
- tests/qcfg/test_failsafe_meta.py
- existing calibration generation tests

## Recommended Execution Order

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

This order minimizes risk by isolating the algorithm first, then model adaptation, then beam search integration, and only after that switching the provider runtime path.
