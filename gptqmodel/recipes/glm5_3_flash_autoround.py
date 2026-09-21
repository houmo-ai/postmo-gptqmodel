# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public AutoRound producer for GLM-5.3-Flash W8 / routed W4 G64.

Optimization uses native decoder blocks and routed, clamped experts. GPTQ is
only the packed checkpoint format; no GPTQ solver or Hessian is involved.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import MethodType

import torch

from .glm5_3_flash import (
    EXPERT_PATTERN,
    Glm53QuantRecipeResult,
    _weight_map,
    _write,
    audit_glm5_3_flash_quantized_artifact,
    expected_glm53_quantized_modules,
    prepare_glm5_3_flash_bf16,
)


def build_glm53_autoround_policy(model):
    """Use AutoRound's regex syntax for experts, exact exclusions elsewhere."""
    expected = expected_glm53_quantized_modules(model.config)
    actual = dict(model.named_modules())
    for name, (_, inputs, outputs) in expected.items():
        module = actual.get(name)
        if not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != (
            outputs,
            inputs,
        ):
            raise ValueError(f"GLM AutoRound projection inventory mismatch: {name}")
    # AutoRound escapes bare dots itself; do not pass already escaped dots.
    policy = {
        r"^model.language_model.layers.\d+.mlp.experts.\d+.(gate_proj|up_proj|down_proj)$": {
            "bits": 4,
            "group_size": 64,
        },
    }
    for name, module in actual.items():
        if type(module) is torch.nn.Linear and name not in expected:
            policy[name] = {"bits": 16, "data_type": "fp"}
    return policy


def build_glm53_autoround_calibration(
    tokenizer,
    *,
    jsonl=None,
    dataset=None,
    data=None,
    text_key="text",
    nsamples=128,
    seqlen=2048,
    seed=42,
):
    """Select full-length, non-repetitive rows and record the exact token input."""
    if data is not None:
        candidates = list(data)
    elif jsonl is not None:
        with Path(jsonl).open(encoding="utf-8") as stream:
            candidates = [json.loads(line) for line in stream if line.strip()]
    elif dataset:
        from datasets import load_dataset

        candidates = list(load_dataset(dataset, split="train"))
    else:
        raise ValueError(
            "AutoRound requires calibration_data, calibration_jsonl or calibration_dataset"
        )
    random.Random(seed).shuffle(candidates)
    rows, inspected, short, repetitive = [], 0, 0, 0
    for candidate in candidates:
        text = candidate if isinstance(candidate, str) else candidate[text_key]
        if not isinstance(text, str):
            raise TypeError("GLM AutoRound calibration rows must contain text strings")
        ids = tokenizer(text, truncation=True, max_length=seqlen)["input_ids"]
        inspected += 1
        if len(ids) < seqlen:
            short += 1
            continue
        if ids.count(ids[-1]) > seqlen // 2:
            repetitive += 1
            continue
        rows.append({"input_ids": ids, "attention_mask": [1] * seqlen})
        if len(rows) == nsamples:
            break
    if len(rows) != nsamples:
        raise ValueError(
            f"GLM AutoRound needs {nsamples} full-length rows of {seqlen} tokens; found {len(rows)}"
        )
    evidence = {
        "actual_nsamples": len(rows),
        "actual_tokens": len(rows) * seqlen,
        "inspected_rows": inspected,
        "dropped_short_rows": short,
        "dropped_repetitive_rows": repetitive,
        "tokenized_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
    }
    if jsonl:
        evidence["jsonl_sha256"] = hashlib.sha256(Path(jsonl).read_bytes()).hexdigest()
    tensors = [
        {key: torch.tensor(value, dtype=torch.long) for key, value in row.items()}
        for row in rows
    ]
    return tensors, evidence


def install_glm53_autoround_loss_guard(autoround):
    original = autoround._get_loss
    evidence = {"loss_evaluations": 0, "minimum_loss": None, "maximum_loss": None}

    def guarded(self, *args, **kwargs):
        loss = original(*args, **kwargs)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                "GLM AutoRound produced non-finite loss; checkpoint publication aborted"
            )
        value = float(loss.detach())
        evidence["loss_evaluations"] += 1
        evidence["minimum_loss"] = (
            value
            if evidence["minimum_loss"] is None
            else min(evidence["minimum_loss"], value)
        )
        evidence["maximum_loss"] = (
            value
            if evidence["maximum_loss"] is None
            else max(evidence["maximum_loss"], value)
        )
        count = evidence["loss_evaluations"]
        if count == 1 or count % getattr(self, "gradient_accumulate_steps", 1) == 0:
            print(
                "GLM AutoRound tuning: "
                f"layer={getattr(self, '_glm53_active_layer', None)} "
                f"loss_evaluations={count} raw_loss={value:.8g}",
                flush=True,
            )
        return loss

    autoround._get_loss = MethodType(guarded, autoround)
    return evidence


def install_glm53_autoround_expert_checkpointing(autoround):
    """Recompute each expert projection's QDQ temporaries during backward.

    A full 288-expert block has billions of FP32 rounding variables. Keeping
    the QDQ autograd intermediates for every routed projection exceeds an
    80 GB card even for one sequence. Non-reentrant checkpointing preserves
    gradients of captured rounding/min-max parameters while retaining only
    projection inputs between forward and backward. Frozen expert matrices
    stay on CPU during tuning, then return to the tuning device before
    unwrapping so quantized-input propagation sees a complete device block.
    """
    from auto_round.wrapper import WrapperLinear
    from torch.utils.checkpoint import checkpoint

    original = autoround.wrapper_block
    evidence = {
        "checkpointed_projections": 0,
        "implementation": "torch_non_reentrant",
        "offloaded_expert_weight_bytes": 0,
    }

    def wrap(*args, **kwargs):
        result = original(*args, **kwargs)
        block = args[0]
        autoround._glm53_active_layer = getattr(
            getattr(block, "self_attn", None), "layer_idx", None
        )
        for name, module in block.named_modules():
            if not isinstance(module, WrapperLinear) or not re.search(
                r"(?:^|\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$", name
            ):
                continue

            def forward(self, inputs, _forward=module.forward):
                if not torch.is_grad_enabled():
                    return _forward(inputs)
                return checkpoint(_forward, inputs, use_reentrant=False)

            module.forward = MethodType(forward, module)

            def unwrapper(self, best_params, _unwrap=module.unwrapper):
                # AutoRound copies final QDQ values into the original layer,
                # preserving its device. Restore it before that copy, including
                # AutoRound's own optional update/offloader lifecycle.
                self.orig_layer.to(self.device)
                return _unwrap(best_params)

            module.unwrapper = MethodType(unwrapper, module)
            evidence["checkpointed_projections"] += 1
            # WrapperLinear._qdq_weight already transfers original weights to
            # its tuning device. Keep these frozen matrices on CPU so they do
            # not compete with FP32 rounding variables and gradients. Biases
            # are absent for this model's expert projections.
            if module.orig_layer.bias is not None:
                raise ValueError("GLM expert offload expects bias-free projections")
            weight = module.orig_layer.weight
            evidence["offloaded_expert_weight_bytes"] += (
                weight.numel() * weight.element_size()
            )
            module.orig_layer.to("cpu")
        return result

    autoround.wrapper_block = wrap
    return evidence


def canonicalize_glm53_autoround_checkpoint(output_dir, *, use_cache=True):
    """Validate native AutoRound rules, then compact the equivalent W4 policy.

    Only metadata is changed. Packed tensors and AutoRound provider/version/
    iteration metadata remain intact. Both metadata entrypoints are updated.
    """
    from transformers import AutoConfig

    root = Path(output_dir)
    config_path = root / "config.json"
    document = json.loads(config_path.read_text())
    expected = expected_glm53_quantized_modules(AutoConfig.from_pretrained(root))
    documents = [(config_path, document, document["quantization_config"])]
    standalone = root / "quantization_config.json"
    if standalone.exists():
        value = json.loads(standalone.read_text())
        documents.append((standalone, value, value.get("quantization_config", value)))
    for path, _, quant in documents:
        if (
            (
                quant.get("provider"),
                quant.get("quant_method"),
                quant.get("bits"),
                quant.get("group_size"),
                quant.get("sym"),
                quant.get("checkpoint_format"),
                quant.get("data_type"),
            )
            != ("auto-round", "gptq", 8, 64, True, "gptq_v2", "int_sym_gptq")
            or quant.get("desc_act", False) is not False
            or int(quant.get("iters", 200)) <= 0
        ):
            fields = (
                "provider", "quant_method", "bits", "group_size", "sym",
                "checkpoint_format", "data_type", "desc_act", "iters",
            )
            actual = {key: quant.get(key) for key in fields}
            raise ValueError(
                f"Unsupported GLM AutoRound packed profile: {path}; actual={actual!r}"
            )
        # Native AutoRound omits defaults, including iters=200, on export.
        # Restore it explicitly for downstream provenance consumers.
        quant.setdefault("iters", 200)
        rules = quant.get("dynamic") or {}
        for name, (bits, _, _) in expected.items():
            effective = {"bits": 8, "group_size": 64, "sym": True}
            for pattern, override in rules.items():
                expression = pattern.removeprefix("+:").removeprefix("-:")
                if re.match(expression, name):
                    if pattern.startswith("-:"):
                        raise ValueError(
                            f"GLM AutoRound excluded a required projection: {name}"
                        )
                    effective.update(override)
                    break
            if (effective["bits"], effective["group_size"], effective["sym"]) != (
                bits,
                64,
                True,
            ):
                raise ValueError(f"GLM AutoRound dynamic precision mismatch: {name}")
        quant["dynamic"] = {
            **{
                pattern: value
                for pattern, value in rules.items()
                if pattern.startswith("-:")
            },
            EXPERT_PATTERN: {"bits": 4, "group_size": 64, "sym": True},
        }
        quant.pop("extra_config", None)
        quant["desc_act"] = False
    document["text_config"]["use_cache"] = bool(use_cache)
    document["use_cache"] = bool(use_cache)
    for path, value, _ in documents:
        _write(path, value)
    return audit_glm5_3_flash_quantized_artifact(root)


def quantize_glm5_3_flash_autoround(
    *,
    model_dir: str,
    output_dir: str,
    prepared_model_dir: str | None = None,
    num_layers: int | None = None,
    method: str = "autoround",
    artifact_format: str = "gptqmodel_hf",
    bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    iters: int = 200,
    nsamples: int = 128,
    seqlen: int = 2048,
    batch_size: int = 1,
    gradient_accumulate_steps: int = 2,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = "NeelNanda/pile-10k",
    calibration_data=None,
    device: str = "cuda:0",
    device_map: str | None = None,
    preparation_devices: tuple[str, ...] = ("cpu",),
    seed: int = 42,
    dry_run: bool = False,
) -> Glm53QuantRecipeResult:
    if (method, artifact_format, bits, expert_bits, group_size, sym) != (
        "autoround",
        "gptqmodel_hf",
        8,
        4,
        64,
        True,
    ):
        raise ValueError(
            "GLM AutoRound requires symmetric W8 / routed W4 G64, gptqmodel_hf format"
        )
    if (
        min(iters, nsamples, seqlen, batch_size, gradient_accumulate_steps) <= 0
        or nsamples % batch_size
        or batch_size * gradient_accumulate_steps > nsamples
    ):
        raise ValueError(
            "AutoRound tuning counts must be positive, nsamples divisible by batch_size, "
            "and the accumulated batch must not exceed nsamples"
        )
    source, output = Path(model_dir).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Quantized output must be outside the source checkpoint")
    prepared = (
        Path(prepared_model_dir).resolve()
        if prepared_model_dir
        else output.with_name(output.name + "_bf16_input")
    )
    selected_devices = device_map if device_map is not None else device
    result = Glm53QuantRecipeResult(
        str(output),
        str(prepared),
        method,
        artifact_format,
        "glm5_3_flash",
        {
            "routed_non_shared_experts": "W4 G64",
            "other_autoround_linears": "W8 G64",
            "deployment_rtn_w8": [
                "vision",
                "lm_head",
                "router",
                "indexer",
                "mhc",
                "conv1d",
                "DSA kv_b factors",
            ],
            "floating_kda": ["forget_gate.f_a_proj", "forget_gate.f_b_proj", "b_proj"],
            "activation": "A16 at deployment",
            "mla_layout": "hf_unabsorbed",
        },
        {
            "nsamples": nsamples,
            "seqlen": seqlen,
            "jsonl": calibration_jsonl,
            "dataset": calibration_dataset,
            "text_key": calibration_text_key,
            "seed": seed,
        },
        {
            "recipe": "gptqmodel.recipes.glm5_3_flash_autoround.quantize_glm5_3_flash_autoround",
            "algorithm": "autoround",
            "packing_format": "auto_round:gptqmodel",
            "dry_run": dry_run,
            "source": str(source),
            "num_layers": num_layers,
            "mtp_included": False,
            "moe_routing": "native_clamped",
            "iters": iters,
            "batch_size": batch_size,
            "gradient_accumulate_steps": gradient_accumulate_steps,
            "device_map": selected_devices,
            "nblocks": 1,
            "low_gpu_mem_usage": True,
            "low_cpu_mem_usage": True,
        },
    )
    if dry_run:
        return result
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    import auto_round
    from auto_round import AutoRound
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        prepare_model_for_moe_quantization,
    )
    from auto_round.utils.model import handle_generation_config
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(source)
    rows, evidence = build_glm53_autoround_calibration(
        tokenizer,
        jsonl=calibration_jsonl,
        dataset=calibration_dataset,
        data=calibration_data,
        text_key=calibration_text_key,
        nsamples=nsamples,
        seqlen=seqlen,
        seed=seed,
    )
    result.calibration.update(evidence)
    prepare_glm5_3_flash_bf16(
        model_dir=str(source),
        output_dir=str(prepared),
        num_layers=num_layers,
        devices=preparation_devices,
    )
    model, loading = AutoModelForImageTextToText.from_pretrained(
        prepared,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        output_loading_info=True,
    )
    invalid = {
        key: loading[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading.get(key)
    }
    if invalid:
        raise ValueError(f"GLM AutoRound source did not load completely: {invalid}")
    # Passing a preloaded module bypasses AutoRound's HF loader, which normally
    # reconciles sampling flags. GLM ships top_p=0.95 without do_sample=True;
    # Transformers rejects saving that combination in strict validation.
    handle_generation_config(model)
    model.generation_config.validate(strict=True)
    if any(kind != "full" for kind in model.config.get_text_config().indexer_types):
        raise ValueError("GLM AutoRound requires independent full DSA indexers")
    use_cache = model.config.get_text_config().use_cache
    model.config.use_cache = model.config.get_text_config().use_cache = False
    prepare_model_for_moe_quantization(model)
    policy = build_glm53_autoround_policy(model)
    blocks = [
        [
            f"model.language_model.layers.{i}"
            for i in range(model.config.get_text_config().num_hidden_layers)
        ]
    ]
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=8,
        group_size=64,
        sym=True,
        data_type="int_sym_gptq",
        act_bits=16,
        dataset=torch.utils.data.DataLoader(rows, batch_size=batch_size, shuffle=False),
        nsamples=nsamples,
        seqlen=seqlen,
        iters=iters,
        batch_size=batch_size,
        gradient_accumulate_steps=gradient_accumulate_steps,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        device_map=selected_devices,
        seed=seed,
        layer_config=policy,
        to_quant_block_names=blocks,
        nblocks=1,
        enable_quanted_input=True,
        enable_torch_compile=False,
    )
    tuning = install_glm53_autoround_loss_guard(autoround)
    expert_checkpointing = install_glm53_autoround_expert_checkpointing(autoround)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.autoround-", dir=output.parent)
    )
    try:
        autoround.quantize_and_save(
            str(staging), format="auto_round:gptqmodel", inplace=True
        )
        if tuning["loss_evaluations"] == 0:
            raise RuntimeError("AutoRound performed no tuning loss evaluations")
        result.provenance.update(
            tuning=tuning,
            expert_checkpointing=expert_checkpointing,
            autoround_version=auto_round.__version__,
            backend_file=auto_round.__file__,
        )
        result.provenance["audit"] = canonicalize_glm53_autoround_checkpoint(
            staging, use_cache=use_cache
        )
        tokenizer.save_pretrained(staging)
        from gptqmodel.utils.model_dequant import copy_aux_files

        copy_aux_files(
            source,
            staging,
            set(_weight_map(source).values())
            | {"config.json", "generation_config.json", "model.safetensors.index.json"},
        )
        _write(staging / "glm53_quantization_recipe.json", asdict(result))
        staging.rename(output)
    except BaseException as error:
        # Tuning may take days. Keep even partial shards for diagnosis/recovery;
        # only the successfully audited directory is published as output.
        error.add_note(f"GLM AutoRound unpublished checkpoint retained at {staging}")
        raise
    finally:
        # Instance callbacks and native wrappers retain bound-method cycles.
        # Release them before a caller starts HMONNX export in this process.
        del autoround, model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return result


__all__ = ["quantize_glm5_3_flash_autoround"]
