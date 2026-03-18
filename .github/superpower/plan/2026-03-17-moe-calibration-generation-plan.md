# MoE Calibration Generation Plan (Minimal Changes)

## Test command baseline
- `source ~/.bashrc && conda activate xhquant && cd /data02/datasets/xuzk_work/gptqmodel/gptqmodel`
- Run pytest target after each task.

## Task 1
1. Add config tests for `calibration_generation`.
2. Implement `CalibrationGenerationConfig` and `QuantizeConfig` parsing/validation/serialization.
3. Run: `pytest -q tests/qcfg/test_calibration_generation_config.py`.

## Task 2
1. Add JSONL loader tests.
2. Implement `gptqmodel/quantization/calibration/generation/jsonl_loader.py`.
3. Run: `pytest -q tests/test_calibration_generation_loader.py`.

## Task 3
1. Add provider registry tests.
2. Implement `gptqmodel/quantization/calibration/generation/base.py`.
3. Run: `pytest -q tests/test_calibration_generation_registry.py`.

## Task 4
1. Add minimal moe_ebss provider test.
2. Implement `gptqmodel/quantization/calibration/generation/moe_ebss_provider.py`.
3. Register provider.
4. Run: `pytest -q tests/test_calibration_generation_moe_ebss_provider.py`.

## Task 5
1. Add quantize hook test for calibration source resolution.
2. Implement `gptqmodel/quantization/calibration/generation/runtime.py`.
3. Hook in `gptqmodel/models/base.py` before mix precision and looper args consumption.
4. Run: `pytest -q tests/test_quantize_calibration_generation_hook.py`.

## Task 6
1. Add minimal example test for config construction.
2. Update `examples/quantization/examples/example_glm_4_7_flash.py` with optional CLI flags and `calibration_generation` wiring; keep default behavior unchanged.
3. Run: `pytest -q tests/test_example_glm_calibration_generation_config.py`.

## Task 7
1. Run all new tests:
   `pytest -q tests/qcfg/test_calibration_generation_config.py tests/test_calibration_generation_loader.py tests/test_calibration_generation_registry.py tests/test_calibration_generation_moe_ebss_provider.py tests/test_quantize_calibration_generation_hook.py tests/test_example_glm_calibration_generation_config.py`
2. Run regression subset:
   `pytest -q tests/qcfg/test_activation.py tests/qcfg/test_failsafe_meta.py`.
