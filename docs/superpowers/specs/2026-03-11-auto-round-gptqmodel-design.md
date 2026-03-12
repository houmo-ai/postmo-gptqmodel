# Auto-Round In GPTQModel Design

## Goal

Add Auto-Round as a new GPTQModel quantization method that:

- reuses GPTQModel's existing checkpoint formats, packing, loading, and qlinear kernels
- fits the current `QuantizeConfig -> Processor -> ModuleLooper -> pack/save` architecture
- stays compatible with GPTQModel-specific features such as rotation, MoE subset routing, and `hessian_mse`-based initialization

## Chosen Approach

Implement Auto-Round as a new `METHOD.AUTO_ROUND` with a dedicated `AutoRoundProcessor`.

The processor will follow the same orchestration model as `AWQProcessor`:

- collect per-module calibration inputs from the existing subset forward path
- cache the float reference outputs needed for local reconstruction loss
- optimize per-module quantization parameters with a short fake-quant loop
- write the final quantized weights plus GPTQModel-compatible `scales`, `zeros`, and `g_idx`
- finalize by reusing existing GPTQ-compatible packing and save/load code paths

This keeps Auto-Round as an algorithm choice rather than a new serialization format.

## Algorithm Shape

The first implementation targets the weight-only Auto-Round path:

- trainable rounding offset `value`
- trainable `max_scale`
- optional `min_scale` for asymmetric quantization
- per-module reconstruction loss between float outputs and fake-quant outputs
- best-parameter tracking across optimization steps

The optimization loop is local to each quantized linear module, which matches GPTQModel's current processor contract better than block-level wrapper training from upstream Auto-Round.

## Hessian-MSE And Initialization

`hessian_mse` will not reuse GPTQ's full sequential error compensation. Instead:

- if disabled, initialize scales from standard min/max or Auto-Round-style searched scales
- if enabled, initialize scales with Hessian-weighted search by reusing GPTQModel's existing scale-solving capability

This keeps the feature useful without coupling Auto-Round to GPTQ's full quantization algorithm.

## Rotation And MoE

Rotation remains a pre-quantization transform handled by `BaseQModel.quantize()`. Auto-Round operates on the already-rotated model, so no processor-specific rotation logic is needed.

For MoE:

- Auto-Round will participate in the existing subset-level capture path
- unrouted experts will use the same skip/failsafe behavior currently applied to AWQ/GPTQ/QQQ processors
- the final validation target is a real MoE quantization run using GPTQModel's existing MoE model support

## Output Format

The new method saves with GPTQModel's existing checkpoint configuration and packed tensors.

No upstream `auto-round` native checkpoint export or import is added in this phase.

## AutoRound v2 Scope

AutoRound v2 support is treated as an extension point:

- add config flags now
- structure the implementation so `init_scale` can be overridden
- attempt `enable_alg_ext` support when it is low-risk

The acceptance bar for this task is still the v1 path plus GPTQModel compatibility features.

## Risks

- current config code contains a hard `group_size == 64` validation that conflicts with existing tests and with Auto-Round requirements
- test coverage currently references Auto-Round symbols that are not implemented, so config and serialization changes must land early
- per-module output reconstruction must stay memory-bounded for MoE-heavy subsets

