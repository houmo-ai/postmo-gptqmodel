# Auto-Round In GPTQModel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Auto-Round as a GPTQModel-native quantization method that reuses existing checkpoint formats and supports rotation, Hessian-MSE scale initialization, tests, and MoE validation.

**Architecture:** Extend GPTQModel's existing method/config pipeline with `METHOD.AUTO_ROUND`, add a dedicated `AutoRoundProcessor`, implement a per-module fake-quant optimization loop that emits GPTQModel-compatible packed tensors, and validate the path with focused unit tests plus a real MoE quantization run.

**Tech Stack:** Python, PyTorch, Transformers, pytest, GPTQModel quantization processors, existing GPTQ qlinear packing

---

## Chunk 1: Config And Loader Plumbing

### Task 1: Add Auto-Round config surface

**Files:**
- Modify: `gptqmodel/quantization/config.py`
- Modify: `gptqmodel/quantization/__init__.py`
- Test: `tests/test_quant_formats_auto_round.py`
- Test: `tests/test_quant_formats_ipex.py`

- [ ] **Step 1: Write the failing config tests**

Add tests that assert:
- `METHOD.AUTO_ROUND` exists
- `AutoRoundQuantizeConfig` exists and round-trips through `to_dict()` / `from_quant_config()`
- Auto-Round keeps GPTQModel checkpoint formats and serializes algorithm-specific fields

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_quant_formats_auto_round.py tests/test_quant_formats_ipex.py -q`
Expected: FAIL because Auto-Round symbols and method plumbing are missing

- [ ] **Step 3: Implement config support**

Add:
- `METHOD.AUTO_ROUND`
- method/format compatibility mapping for GPTQModel-native formats
- `AutoRoundQuantizeConfig`
- serialization and deserialization of Auto-Round-specific fields
- removal of the hard `group_size == 64` restriction in favor of valid positive group sizes

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/test_quant_formats_auto_round.py tests/test_quant_formats_ipex.py -q`
Expected: PASS or only fail on later missing runtime plumbing

- [ ] **Step 5: Commit**

Run: `git add gptqmodel/quantization/config.py gptqmodel/quantization/__init__.py tests/test_quant_formats_auto_round.py tests/test_quant_formats_ipex.py && git commit -m "feat: add autoround quantization config"`

### Task 2: Register Auto-Round in load and quant kernel selection

**Files:**
- Modify: `gptqmodel/models/base.py`
- Modify: `gptqmodel/models/auto.py`
- Modify: `gptqmodel/utils/importer.py`
- Test: `tests/test_quant_formats_auto_round.py`

- [ ] **Step 1: Write or extend failing loader tests**

Add assertions that Auto-Round checkpoints are recognized as supported quantization configs and select GPTQ-compatible qlinear kernels.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_quant_formats_auto_round.py -q`
Expected: FAIL because loader/importer do not recognize Auto-Round

- [ ] **Step 3: Implement loader and importer support**

Update:
- quantization support checks in `models/auto.py`
- `BaseQModel.quantize()` processor selection
- qlinear method compatibility declarations so GPTQ-compatible kernels accept Auto-Round

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/test_quant_formats_auto_round.py -q`
Expected: PASS or fail only on missing processor implementation

- [ ] **Step 5: Commit**

Run: `git add gptqmodel/models/base.py gptqmodel/models/auto.py gptqmodel/utils/importer.py tests/test_quant_formats_auto_round.py && git commit -m "feat: register autoround quantization method"`

## Chunk 2: Auto-Round Algorithm Core

### Task 3: Add low-level Auto-Round quantization helpers

**Files:**
- Create: `gptqmodel/quantization/auto_round.py`
- Test: `tests/test_auto_round_quantizer.py`

- [ ] **Step 1: Write failing quantizer tests**

Add tests for:
- symmetric and asymmetric fake quantization
- group-wise scale shapes
- best-parameter restoration
- Hessian-MSE initialization producing stable scales

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_auto_round_quantizer.py -q`
Expected: FAIL because helper module is missing

- [ ] **Step 3: Implement minimal Auto-Round helper types**

Add:
- fake-quant functions for sym/asym group-wise quantization
- trainable state container for `value`, `min_scale`, `max_scale`
- initialization helpers, including Hessian-weighted scale init
- best-state snapshot/restore utilities

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/test_auto_round_quantizer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add gptqmodel/quantization/auto_round.py tests/test_auto_round_quantizer.py && git commit -m "feat: add autoround quantization helpers"`

### Task 4: Add AutoRoundProcessor

**Files:**
- Create: `gptqmodel/looper/auto_round_processor.py`
- Modify: `gptqmodel/looper/stage_subset.py`
- Modify: `gptqmodel/looper/stage_layer.py`
- Test: `tests/test_auto_round_processor.py`

- [ ] **Step 1: Write failing processor tests**

Add tests for:
- per-module activation capture
- reconstruction loss optimization updates weights
- GPTQModel-compatible `scales/zeros/g_idx` are emitted
- MoE skip logic includes Auto-Round processor

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_auto_round_processor.py -q`
Expected: FAIL because processor is missing

- [ ] **Step 3: Implement AutoRoundProcessor**

Model it after `AWQProcessor`:
- clone per-module config with dynamic overrides
- capture inputs and float outputs
- run local optimization loop on target device
- store final qparams in module state
- finalize through existing quant module creation and pack flow

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/test_auto_round_processor.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add gptqmodel/looper/auto_round_processor.py gptqmodel/looper/stage_subset.py gptqmodel/looper/stage_layer.py tests/test_auto_round_processor.py && git commit -m "feat: add autoround processor"`

## Chunk 3: Compatibility And Integration

### Task 5: Cover rotation and Hessian-MSE integration

**Files:**
- Modify: `tests/test_auto_round_processor.py`
- Create: `tests/test_auto_round_rotation.py`

- [ ] **Step 1: Write failing integration tests**

Add tests that:
- run Auto-Round on a rotated toy module path
- verify `hessian_mse=True` changes scale initialization path without breaking quantization

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_auto_round_processor.py tests/test_auto_round_rotation.py -q`
Expected: FAIL before full integration is complete

- [ ] **Step 3: Implement missing integration glue**

Wire any missing config propagation or helper calls so rotation and Hessian-based initialization work cleanly in the processor path.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `pytest tests/test_auto_round_processor.py tests/test_auto_round_rotation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add tests/test_auto_round_processor.py tests/test_auto_round_rotation.py gptqmodel/looper/auto_round_processor.py gptqmodel/quantization/auto_round.py && git commit -m "feat: integrate autoround with rotation and hessian init"`

### Task 6: Add MoE validation coverage

**Files:**
- Create: `tests/models/auto_round/test_moe.py`
- Modify: any helper files only if required

- [ ] **Step 1: Write the failing MoE test**

Model it after existing AWQ MoE tests and assert:
- Auto-Round quantization completes on a supported MoE model
- generated or evaluated output clears a minimal correctness bar

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/models/auto_round/test_moe.py -q`
Expected: FAIL before MoE path is fully wired

- [ ] **Step 3: Implement missing MoE-specific fixes**

Update Auto-Round processor behavior only where required for subset routing, skipped experts, or finalization safety.

- [ ] **Step 4: Run the targeted test to verify it passes**

Run: `pytest tests/models/auto_round/test_moe.py -q`
Expected: PASS in the configured environment

- [ ] **Step 5: Commit**

Run: `git add tests/models/auto_round/test_moe.py gptqmodel/looper/auto_round_processor.py gptqmodel/models/base.py && git commit -m "test: validate autoround on moe models"`

## Chunk 4: End-To-End Verification

### Task 7: Run focused verification and real MoE quantization

**Files:**
- Modify: none unless verification exposes defects

- [ ] **Step 1: Run focused unit and integration tests**

Run: `pytest tests/test_auto_round_quantizer.py tests/test_auto_round_processor.py tests/test_auto_round_rotation.py tests/test_quant_formats_auto_round.py tests/test_quant_formats_ipex.py -q`
Expected: PASS

- [ ] **Step 2: Run the real MoE validation**

Run: `pytest tests/models/auto_round/test_moe.py -q`
Expected: PASS

- [ ] **Step 3: Fix any issues revealed by verification**

Apply the minimal code changes needed, then rerun the exact failing command until clean.

- [ ] **Step 4: Run a final combined verification pass**

Run: `pytest tests/test_auto_round_quantizer.py tests/test_auto_round_processor.py tests/test_auto_round_rotation.py tests/test_quant_formats_auto_round.py tests/test_quant_formats_ipex.py tests/models/auto_round/test_moe.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add -A && git commit -m "feat: add autoround quantization to gptqmodel"`

