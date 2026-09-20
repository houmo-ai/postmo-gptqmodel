# Qwen3.5 / Qwen3.6 MoE automatic block precision

Both public recipes accept `auto_mixed_precision=True` or a strict mapping:

```python
from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

result = quantize_qwen35_autoround(
    model_dir="/models/Qwen3.6-35B-A3B",
    output_dir="/outputs/qwen36-auto-mixed",
    topology="moe", bits=4, expert_bits=4, expert_down_bits=4,
    data_type="int_sym_xh", format="auto_round:gptqmodel",
    dataset="/data/search-and-calibration.jsonl",
    nsamples=256, seqlen=2048, iters=400, batch_size=8, seed=42,
    head={"bits": 5},
    auto_mixed_precision={
        "candidates": [4, 5, 8], "initial_bits": 5,
        "max_kl": 0.03, "max_ppl_increase": 0.02,
        "nsamples": 32, "seqlen": 512, "positions_per_sample": 64,
        "max_trials": 24, "seed": 42, "device": "cuda:0",
        "image_dataset": "/data/image-search.jsonl", "image_nsamples": 64,
        "image_seqlen": 2048, "image_max_pixels": 393216,
    },
)
```

This single call performs the complete pipeline: select W4/W5/W8 block bits,
restore the floating weights, calibrate the backbone with AutoRound using the
selected bits, then calibrate Head W5 from that quantized backbone's activations.
The head is calibrated once from its floating weights, after the backbone; no
intermediate Head W4 conversion or separate user invocation is required. Routed
experts remain W4 throughout. The result's `output_dir` is the final checkpoint,
including the bit map and calibrated head; the temporary backbone is removed.
`quantize_qwen35` follows the same sequence with GPTQ for both calibration stages.

The lightweight profile uses KL 0.03 and sampled PPL increase 0.02, with 32 text
samples and 64 short-answer images for search. Body and head calibration use the
separately configured text dataset (256 samples, length 2048, 400 iterations in
this AutoRound example). These search budgets are not a bound on task accuracy.

`quantize_qwen35` accepts the same search mapping. The final algorithm follows
which recipe was called. Omitted/False preserves the existing flow. An empty
mapping means enabled defaults; unknown keys are errors. `dataset`, `text_key`
and `device` can override the search source, text field and single GPU. Search
otherwise reuses the final calibration source, but has its own sample count,
length and seed. Only the train split of a named HF text dataset or a local
JSONL is supported for text. Use a held-out dataset for acceptance.

For multimodal search, `image_dataset` selects a local JSONL with one image,
prompt and reference answer per record:

```json
{"image": "images/example.png", "text": "What is written on the sign?", "answer": "Library"}
```

Image paths are relative to the JSONL (absolute local paths are also accepted).
For a self-contained JSONL, replace `image` with `image_base64` containing raw
Base64 of the original image file bytes (no data-URI prefix):

```json
{"image_base64": "<base64 image bytes>", "text": "What is written on the sign?", "answer": "Library"}
```

Each record must provide exactly one of `image` and `image_base64`. Optional
`image_sha256` is a 64-digit hexadecimal SHA256 of the original encoded file
bytes, checked for either form before decoding. `image_name` may retain the
original filename as metadata; it is not opened. Inline images are decoded in
memory, require no sidecar files or download, and use the same preprocessing
as path images. Invalid Base64, undecodable images and digest mismatches fail.
The native processor builds the image patches and chat prompt with thinking
disabled. Only reference-answer tokens are scored; image placeholders and the
question are excluded. Long image sequences raise instead of silently truncating
patches. `image_seqlen` bounds the complete prompt plus answer; `image_max_pixels`
bounds resized image area. Missing/malformed requested image data fails explicitly.
`image_nsamples` defaults to 32. Omit `image_dataset` for text-only screening.
Optional `options` lists single-token answer labels (or maps labels to option
text); the prompt must already contain the options. The report includes correct
counts and choice agreement as diagnostics, not an additional accuracy gate.

Each text block has two independent units: attention main projections and the
shared expert gate/up/down. Full attention q/k/v/o share bits; linear attention
qkv/z/out share bits. Other weights, including routed experts, routers, gates,
a/b projections, convolution, vision and head, are excluded. Routed expert
bit overrides remain the caller's responsibility. Layer count is discovered;
the 35B-A3B model has 40 layers, 80 units and 250 target Linear modules.

Search uses symmetric group RTN QDQ with positive FP32 scales as a fast
surrogate. Reference activations rank candidates by normalized Linear-output
reconstruction error. Native floating full-model logits are the reference.
All non-search weights remain floating and routing stays native in screening.
Both text and image inputs contribute to reconstruction ranking. Each image
runs through the real native vision tower and M-RoPE during the reference pass;
the unchanged vision embeddings and positions are cached for later full-language
model replay. Tests compare this cache path with uncached native image forwards.
The all-W8 control is evaluated separately before the budgeted search.
The initial all-W5 proposal is measured globally. If infeasible, greedy W8
fallback batches recover a feasible assignment. Batch size ensures all W8 can
be evaluated within budget when at least two trials are available. Feasible
plans then undo unnecessary W8 promotions in batches, bisecting rejected batches.
Previously measured assignments reuse their metrics. Remaining trials try successive lower
bits, ordered by predicted error per logical bit saved. Every acceptance uses
actual cumulative full-model output metrics, never a sum of per-block KL.
This bounded greedy procedure does not promise a globally optimal assignment.

Metric semantics: full vocabulary `KL(P_float || Q_candidate)`, natural logs,
temperature 1, FP32 log-softmax, identical teacher-forced input history. Select
uniform valid next-token prediction positions, mask padding and the last
position, weight samples by their scored token counts. The PPL increase is
`exp(mean(NLL_candidate - NLL_reference)) - 1` on those same sampled positions.
Text and image-answer KL/PPL must **each** satisfy `max_kl` and
`max_ppl_increase`. They are never averaged together. With images enabled,
`metrics.modalities` contains separate results; the top-level KL/PPL fields
are the maximum across modalities. Text-rejected proposals skip image replay
and record `skipped_modalities`; accepted proposals always have both results.
The report's `max_sample_p99_kl` is the largest per-sample 99th percentile,
not the global token percentile. Thresholds gate screening only, not final
AutoRound/GPTQ, multimodal task accuracy, long-context/decode or A8 deployment.
Image scoring is teacher-forced answer evaluation, not free generation or an
independent multimodal benchmark. Default thresholds are a starting profile,
not a universally validated accuracy bound. Final solver calibration continues
to use its separately configured calibration dataset.

All changed weights, hooks, training flags and the original device are restored
before final calibration and also on search errors. Native floating input is
required; existing packed models, rotation, disk offload and partial-layer
quantization are rejected. Search currently moves the native model to one GPU;
this is independent of final AutoRound device mapping/memory options. Use an
80 GB GPU for the 35B-A3B model. Models larger than single-GPU capacity need a
future streaming search executor.

Final calibration uses the selected static module map. GPTQ exact rules precede
broad regex rules because matching is first-hit. Local W5 has 5-bit numerical
range in 8-bit slots; savings in the report count logical compiler bits, not
checkpoint file size. The final checkpoint includes `auto_mixed_precision.json`
and `quantization_config.meta.auto_mixed_precision` in standard config files.
The reader verifies complete block coverage, per-module bits and packed tensor
shapes. Head completion preserves the plan and full report. A sibling
`<output_dir>.auto_mixed_precision_search.json` preserves failed/successful
search evidence, including when the body was built in a temporary directory.
No feasible assignment within budget raises instead of quantizing a failed plan.

Validation uses a reduced-dimension native hybrid-MoE fixture (not pretrained
weights): CPU metric/config/state contracts, real AutoRound and GPTQ calibration,
W4/W5/W8 packed save/reload, tied head W4 completion, and native loaded forward.
ModelZoo owns the YAML/recipe and deployment-consumption tests.
