# MoE Calibration Generation Design

## Goal

Make calibration dataset generation a built-in part of the current quantization toolchain, controlled by config.
When enabled, use the existing moe_ebss capability under quantization calibration to generate calibration data, then use the generated dataset for subsequent quantization calibration.

## Scope

- Entry layer: QuantizeConfig
- Model scope (v1): generic interface for all MoE models
- Data form: JSONL on disk
- Failure policy: fail-fast (stop quantization immediately)

## Non-goals (v1)

- Silent fallback to public/default calibration datasets
- Reworking existing downstream quantization loop behavior
- Supporting dense-only generation algorithms in this change

## Architecture

### 1) Config Extension

Add a new optional config section in QuantizeConfig:

- calibration_generation.enabled: bool
- calibration_generation.provider: str (default: moe_ebss)
- calibration_generation.output_path: str
- calibration_generation.nsamples: int
- calibration_generation.seqlen: int
- calibration_generation.num_beams: int
- calibration_generation.seed: int | null
- calibration_generation.overwrite: bool
- calibration_generation.model_patch: str | null
- calibration_generation.allow_override: bool (controls precedence with user-provided calibration input)

Validation rules:

- If enabled=True, provider and output_path are required
- nsamples, seqlen, num_beams must be positive
- provider must be a registered provider name

### 2) Execution Hook Point

In quantize pre-processing stage:

1. Read QuantizeConfig.calibration_generation
2. If enabled and provider=moe_ebss, run generation provider
3. Ensure JSONL exists and has valid rows
4. Load JSONL into calibration source
5. Pass loaded source into existing prepare_calibration_dataset pipeline

No behavior changes for quantization steps after dataset preparation.

### 3) Provider-based Generation Layer

Introduce a provider registry abstraction:

- Base protocol: generate(model, tokenizer, cfg) -> GeneratedDatasetArtifact
- Registry: provider name -> implementation class

Initial provider:

- MoeEbssProvider: adapts current moe_ebss generation logic

This keeps future extensibility for additional generation strategies.

### 4) Data Loading Layer

JSONL loader responsibilities:

- Parse generated JSONL rows (default key: text)
- Validate non-empty rows and basic text integrity
- Convert to calibration input shape already accepted by prepare_calibration_dataset

## Component Plan

- quantization/config.py
  - Add calibration_generation dataclass/schema and validation
- new: quantization/calibration/generation/base.py
  - provider protocol + registry
- new: quantization/calibration/generation/moe_ebss_provider.py
  - moe_ebss generation implementation
- new: quantization/calibration/generation/jsonl_loader.py
  - JSONL -> calibration input conversion
- models/base.py
  - hook pre-quantization generation-and-load path
- examples/quantization/examples/example_glm_4_7_flash.py
  - add optional usage demo via QuantizeConfig

## Data Flow

1. User creates QuantizeConfig with calibration_generation.enabled=True
2. quantize() starts
3. generation provider runs and writes JSONL
4. JSONL loader reads rows and converts to calibration source
5. prepare_calibration_dataset tokenizes/packs batches
6. quantization proceeds unchanged

## Error Handling

Fail-fast for these cases:

- Invalid generation config
- Unsupported model/provider mismatch
- Generation runtime failure
- Output file missing/empty/corrupted
- Loader conversion failure

All above should raise explicit errors with actionable messages.

## Compatibility and Precedence

- If calibration_generation.enabled=False: current behavior remains unchanged
- If enabled=True and external calibration is provided:
  - default behavior controlled by allow_override
  - recommended default: external calibration has priority unless allow_override=True

## Observability

Log key lifecycle events:

- Provider name and resolved config
- Generation start/end, elapsed time
- Output path and generated sample count
- Loader accepted/filtered row counts

## Testing Strategy

### Unit tests

- config parsing and validation
- provider registry lookup
- provider model capability dispatch
- JSONL loader validation and conversion

### Integration tests

- enabled flow: generate -> load -> prepare_calibration_dataset
- fail-fast path on generation error
- behavior parity when disabled

### Regression checks

- Existing quantization path without generation remains unaffected

## Rollout

Phase 1:

- Land config + provider abstraction + moe_ebss provider + loader + quantize hook
- Add minimal example usage

Phase 2:

- Expand model patch routing and provider capabilities
- Add richer generation diagnostics/metrics

## Acceptance Criteria

- User can enable calibration generation purely via QuantizeConfig
- Generated JSONL is consumed automatically in the same quantize run
- On generation failure, quantization terminates with clear error
- Existing non-generation quantization flows pass regression tests
