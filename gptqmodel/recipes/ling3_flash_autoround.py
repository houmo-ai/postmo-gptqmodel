"""Ling AutoRound engine; called by the public ling3_flash recipe.

Layer selection, calibration, optimization guards and GPTQ metadata belong to
the quantization backend. This module does not depend on ModelZoo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MethodType

import torch


def _is_kda_decay_or_beta_projection(name: str) -> bool:
    return name.endswith(
        (
            ".attention.f_proj",
            ".attention.f_a_proj",
            ".attention.f_b_proj",
            ".attention.b_proj",
        )
    )


def _ling_layer_layout(
    model: torch.nn.Module,
) -> tuple[list[str], list[str]]:
    """Return decoder and MTP block names without relying on a fixed prefix."""

    config = getattr(model, "config", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    language_model = getattr(model, "model", None)
    layers = getattr(language_model, "layers", None)
    if num_hidden_layers is None or layers is None:
        raise RuntimeError(
            "Ling model is missing config.num_hidden_layers or model.layers"
        )

    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers <= 0 or len(layers) < num_hidden_layers:
        raise RuntimeError(
            "Ling layer layout is invalid: "
            f"num_hidden_layers={num_hidden_layers}, total_layers={len(layers)}"
        )

    layer_prefixes = [
        name for name, module in model.named_modules() if module is layers
    ]
    if len(layer_prefixes) != 1 or not layer_prefixes[0]:
        raise RuntimeError(
            f"Unable to resolve the Ling model.layers prefix: {layer_prefixes}"
        )
    layer_prefix = layer_prefixes[0]
    decoder_blocks = [f"{layer_prefix}.{index}" for index in range(num_hidden_layers)]
    mtp_blocks = [
        f"{layer_prefix}.{index}" for index in range(num_hidden_layers, len(layers))
    ]
    return decoder_blocks, mtp_blocks


def build_autoround_block_names(model: torch.nn.Module) -> list[list[str]]:
    """Select only normal decoder blocks; MTP has a two-input forward contract."""

    decoder_blocks, _ = _ling_layer_layout(model)
    return [decoder_blocks]


def _build_autoround_regex_policy(
    decoder_blocks: list[str],
    mtp_blocks: list[str],
    *,
    expert_bits: int,
    group_size: int,
) -> dict[str, dict[str, int | str]]:
    """Build non-overlapping regex rules in AutoRound's input syntax."""

    if not decoder_blocks:
        raise RuntimeError("Ling AutoRound policy requires decoder blocks")
    all_blocks = [*decoder_blocks, *mtp_blocks]
    if any(re.search(r"[^A-Za-z0-9_.]", name) for name in all_blocks):
        raise RuntimeError(
            f"Ling block names cannot be represented safely: {all_blocks}"
        )

    decoder_group = "(?:" + "|".join(decoder_blocks) + ")"
    policy: dict[str, dict[str, int | str]] = {}
    if mtp_blocks:
        mtp_group = "(?:" + "|".join(mtp_blocks) + ")"
        policy[f"^{mtp_group}(?:[.].*)$"] = {
            "bits": 16,
            "data_type": "fp",
        }
    policy[
        f"^{decoder_group}.attention."
        r"(?:f_proj|f_a_proj|f_b_proj|b_proj)$"
    ] = {"bits": 16, "data_type": "fp"}
    policy[
        f"^{decoder_group}.mlp.experts."
        r"\d+.(?:gate_proj|up_proj|down_proj)$"
    ] = {"bits": expert_bits, "group_size": group_size}
    return policy


def build_autoround_layer_config(
    model: torch.nn.Module,
    *,
    expert_bits: int = 4,
    group_size: int = 64,
) -> dict[str, dict[str, int | str]]:
    """Return three regex policies instead of enumerating every expert Linear.

    Ling's f/b projections correspond to Qwen3.5's excluded
    ``in_proj_a/in_proj_b`` decay/beta path. Every other Linear inherits the
    AutoRound base W8/G64/symmetric scheme. Supplying regexes here is
    important: AutoRound preserves them in ``regex_config`` and can collapse
    the expanded per-module state when exporting GPTQModel metadata.
    """

    decoder_blocks, mtp_blocks = _ling_layer_layout(model)
    decoder_prefixes = tuple(f"{name}." for name in decoder_blocks)
    mtp_prefixes = tuple(f"{name}." for name in mtp_blocks)
    routed_experts = 0
    excluded_kda = 0
    excluded_mtp = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.startswith(mtp_prefixes):
            excluded_mtp += 1
        elif name.startswith(decoder_prefixes) and _is_kda_decay_or_beta_projection(
            name
        ):
            excluded_kda += 1
        elif (
            name.startswith(decoder_prefixes)
            and ".mlp.experts." in name
            and name.endswith((".gate_proj", ".up_proj", ".down_proj"))
        ):
            routed_experts += 1
    if routed_experts == 0:
        raise RuntimeError("AutoRound found no routed expert Linear modules")
    if excluded_kda == 0:
        raise RuntimeError(
            "AutoRound found no Ling KDA decay/beta projections to exclude"
        )
    if mtp_blocks and excluded_mtp == 0:
        raise RuntimeError("AutoRound found Ling MTP blocks but no MTP Linear modules")
    return _build_autoround_regex_policy(
        decoder_blocks,
        mtp_blocks,
        expert_bits=expert_bits,
        group_size=group_size,
    )


def _autoround_device_map(value: str):
    value = str(value).strip()
    if value.isdigit():
        return int(value)
    if "," in value and all(part.strip().isdigit() for part in value.split(",")):
        raise ValueError(
            "Pass one local AutoRound GPU index per experiment, for example device_map='0'"
        )
    return value


def _install_autoround_nonfinite_loss_guard(autoround) -> None:
    """Fail immediately instead of exporting after an invalid tuning loss."""

    original_get_loss = autoround._get_loss

    def _guarded_get_loss(self, *args, **kwargs):
        loss = original_get_loss(*args, **kwargs)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                "Ling-3-Flash AutoRound produced a non-finite tuning loss. "
                "The quantized checkpoint was not exported."
            )
        return loss

    autoround._get_loss = MethodType(_guarded_get_loss, autoround)


def canonicalize_autoround_gptqmodel_config(output_dir: str | Path) -> None:
    """Validate native regex export and remove redundant expanded metadata.

    ``auto_round:gptqmodel`` consumers use ``dynamic``. AutoRound also writes
    its expanded ``extra_config`` for compatibility with its native loader;
    that data is redundant because the same loader can expand ``dynamic``.
    Removing it keeps Ling's configuration proportional to the policy rather
    than to its 512-expert module count. AutoRound writes the metadata both
    inside ``config.json`` and, on some versions, into a standalone
    ``quantization_config.json``. Canonicalize both so loaders cannot
    accidentally select the still-expanded copy.
    """

    config_path = Path(output_dir).expanduser().resolve() / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise RuntimeError(
            f"AutoRound output has no quantization_config: {config_path}"
        )
    dynamic = quantization_config.get("dynamic")
    if not isinstance(dynamic, dict) or not dynamic:
        raise RuntimeError(f"AutoRound output has no dynamic rules: {config_path}")

    if (
        int(quantization_config.get("bits", 0)) != 8
        or int(quantization_config.get("group_size", 0)) != 64
        or quantization_config.get("sym") is not True
    ):
        raise RuntimeError(
            "Ling AutoRound output is not the required W8/G64/sym profile"
        )

    num_hidden_layers = int(config.get("num_hidden_layers", 0))
    num_mtp_layers = int(config.get("num_nextn_predict_layers", 0))
    if num_hidden_layers <= 0 or num_mtp_layers < 0:
        raise RuntimeError(
            "Ling AutoRound output must declare positive num_hidden_layers and non-negative "
            "num_nextn_predict_layers"
        )
    block_prefixes = quantization_config.get("block_name_to_quantize")
    if not (
        isinstance(block_prefixes, list)
        and len(block_prefixes) == 1
        and isinstance(block_prefixes[0], str)
    ):
        raise RuntimeError(
            "Ling AutoRound output must contain one block_name_to_quantize prefix"
        )
    layer_prefix = block_prefixes[0]
    decoder_blocks = [f"{layer_prefix}.{index}" for index in range(num_hidden_layers)]
    mtp_blocks = [
        f"{layer_prefix}.{index}"
        for index in range(num_hidden_layers, num_hidden_layers + num_mtp_layers)
    ]
    policy = _build_autoround_regex_policy(
        decoder_blocks,
        mtp_blocks,
        expert_bits=4,
        group_size=64,
    )

    from auto_round.utils import to_standard_regex

    expected_dynamic: dict[str, dict[str, int]] = {}
    expected_regex_values: dict[str, dict[str, int | str]] = {}
    for pattern, layer_cfg in policy.items():
        standardized = to_standard_regex(pattern)
        expected_regex_values[standardized] = layer_cfg
        if int(layer_cfg["bits"]) >= 16:
            expected_dynamic[f"-:{standardized}"] = {}
        else:
            expected_dynamic[f"+:{standardized}"] = {"bits": int(layer_cfg["bits"])}

    def resolve(rules: dict, module_name: str):
        for pattern, overrides in rules.items():
            if pattern.startswith("-:"):
                if re.match(pattern.removeprefix("-:"), module_name):
                    return False
            elif re.match(pattern.removeprefix("+:"), module_name):
                return overrides
        return None

    # AutoRound also excludes Linear shapes that GPTQ packing cannot represent.
    # Retain only exact exclusions backed by an actual unpackable float tensor;
    # do not trust arbitrary bits=16 metadata or silently discard dynamic rules.
    packing_exclusions = {}
    exact_exclusion = re.compile(
        r"^-:\^(?P<name>[A-Za-z0-9_]+(?:\\\.[A-Za-z0-9_]+)*)\$$"
    )
    for pattern, overrides in dynamic.items():
        match = exact_exclusion.fullmatch(pattern)
        if match is None:
            continue
        name = match.group("name").replace(r"\.", ".")
        if resolve(expected_dynamic, name) is False:
            continue
        if overrides != {} or not name.startswith(
            tuple(f"{block}." for block in decoder_blocks)
        ):
            raise RuntimeError(f"Invalid AutoRound packing exclusion: {pattern}")
        packing_exclusions[name] = pattern
    if packing_exclusions:
        from safetensors import safe_open

        shapes = {}
        for shard in sorted(config_path.parent.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as weights:
                keys = set(weights.keys())
                for name in packing_exclusions:
                    if f"{name}.qweight" in keys:
                        raise RuntimeError(
                            f"Excluded AutoRound module has packed weights: {name}"
                        )
                    if f"{name}.weight" in keys:
                        tensor = weights.get_slice(f"{name}.weight")
                        if tensor.get_dtype() not in {"F16", "BF16", "F32"}:
                            raise RuntimeError(
                                f"AutoRound packing exclusion is not floating point: {name}"
                            )
                        shapes[name] = tensor.get_shape()
        for name, pattern in packing_exclusions.items():
            shape = shapes.get(name)
            if shape is None or len(shape) != 2 or all(dim % 32 == 0 for dim in shape):
                raise RuntimeError(
                    f"AutoRound packing exclusion has no unpackable Linear weight: {name}"
                )
        expected_dynamic = {
            **{pattern: {} for pattern in packing_exclusions.values()},
            **expected_dynamic,
        }

    def validate_and_canonicalize(candidate: dict, *, source: Path) -> None:
        candidate_dynamic = candidate.get("dynamic")
        if not isinstance(candidate_dynamic, dict) or not candidate_dynamic:
            raise RuntimeError(f"Ling AutoRound config has no dynamic rules: {source}")
        for field in (
            "bits",
            "group_size",
            "sym",
            "block_name_to_quantize",
            "data_type",
            "checkpoint_format",
            "quant_method",
        ):
            if candidate.get(field) != quantization_config.get(field):
                raise RuntimeError(
                    f"Ling AutoRound metadata mismatch for {field!r} in {source}: "
                    f"expected {quantization_config.get(field)!r}, got {candidate.get(field)!r}"
                )

        extra_config = candidate.get("extra_config", {})
        if extra_config is not None and not isinstance(extra_config, dict):
            raise RuntimeError(
                f"Ling AutoRound extra_config must be a mapping: {source}"
            )

        exact_entries = 0
        for name, layer_cfg in (extra_config or {}).items():
            if name in expected_regex_values:
                expected = (
                    False
                    if int(expected_regex_values[name]["bits"]) >= 16
                    else {"bits": int(expected_regex_values[name]["bits"])}
                )
            else:
                expected = resolve(expected_dynamic, name)
                exact_entries += 1
            if expected is False:
                valid = int(layer_cfg.get("bits", 16)) >= 16
            elif isinstance(expected, dict):
                valid = int(layer_cfg.get("bits", 8)) == int(expected["bits"])
            else:
                valid = False
            if not valid:
                raise RuntimeError(
                    f"Unexpected Ling AutoRound extra_config entry in {source}: "
                    f"{name!r}: {layer_cfg!r}"
                )

        if extra_config and exact_entries == 0:
            raise RuntimeError(
                f"Ling AutoRound extra_config did not contain expanded module state: {source}"
            )

        if candidate_dynamic != expected_dynamic:
            # Validate every rule, even for small checkpoints. Checking only
            # expanded entries could silently erase an unknown compact rule.
            exact_rule = re.compile(
                r"^(?P<kind>[+-]):\^(?P<name>[A-Za-z0-9_]+(?:\\\.[A-Za-z0-9_]+)*)\$$"
            )
            for pattern, overrides in candidate_dynamic.items():
                if pattern in expected_dynamic:
                    expected = expected_dynamic[pattern]
                    actual = overrides
                    name = pattern
                else:
                    match = exact_rule.fullmatch(pattern)
                    if match is None:
                        raise RuntimeError(
                            f"Unexpected expanded Ling AutoRound dynamic rule in {source}: "
                            f"{pattern!r}"
                        )
                    name = match.group("name").replace(r"\.", ".")
                    expected = resolve(expected_dynamic, name)
                    actual = False if match.group("kind") == "-" else overrides
                if actual != expected:
                    raise RuntimeError(
                        f"Ling AutoRound dynamic mismatch for {name!r} in {source}: "
                        f"expected {expected!r}, got {actual!r}"
                    )
            if not extra_config:
                raise RuntimeError(
                    f"Cannot verify expanded Ling AutoRound rules without extra_config: {source}"
                )
            for name in extra_config:
                if name in expected_regex_values:
                    continue
                if resolve(candidate_dynamic, name) != resolve(expected_dynamic, name):
                    raise RuntimeError(
                        f"Ling AutoRound dynamic mismatch for expanded module {name!r} in {source}"
                    )

        candidate["dynamic"] = expected_dynamic
        candidate.pop("extra_config", None)

    validate_and_canonicalize(quantization_config, source=config_path)

    documents_to_write = [(config_path, config)]
    standalone_path = config_path.with_name("quantization_config.json")
    if standalone_path.is_file():
        standalone_document = json.loads(standalone_path.read_text(encoding="utf-8"))
        standalone_quantization_config = standalone_document.get(
            "quantization_config", standalone_document
        )
        if not isinstance(standalone_quantization_config, dict):
            raise RuntimeError(
                f"AutoRound standalone quantization metadata is not a mapping: {standalone_path}"
            )
        validate_and_canonicalize(
            standalone_quantization_config,
            source=standalone_path,
        )
        documents_to_write.append((standalone_path, standalone_document))

    temporary_documents = []
    for path, document in documents_to_write:
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_documents.append((temporary_path, path))
    for temporary_path, path in temporary_documents:
        temporary_path.replace(path)


def _calibration_loader(source, tokenizer, *, seqlen, nsamples, batch_size, seed):
    if isinstance(source, str):
        return source
    from auto_round.calib_dataset import get_tokenizer_function
    from torch.utils.data import DataLoader

    dataset = source.shuffle(seed=seed).map(
        get_tokenizer_function(tokenizer, seqlen),
        batched=True,
    )

    def usable(row):
        ids = row["input_ids"]
        return len(ids) >= seqlen and ids[:seqlen].count(ids[seqlen - 1]) <= seqlen // 2

    dataset = dataset.filter(usable)
    if len(dataset) < nsamples:
        raise ValueError(
            f"Only {len(dataset)} full-length calibration rows, need {nsamples}"
        )
    dataset = dataset.select(range(nsamples))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def quantize_autoround(
    *,
    model_dir: str,
    output_dir: str,
    calibration_source,
    bits: int,
    expert_bits: int,
    group_size: int,
    data_type: str,
    nsamples: int,
    seqlen: int,
    iters: int,
    batch_size: int,
    device_map: str,
    seed: int,
    trust_remote_code: bool,
    mla_weight_layout: str = "hf_unabsorbed",
) -> None:
    if trust_remote_code is not True:
        raise ValueError("Ling AutoRound requires trust_remote_code=True for the checkpoint-side model code")
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.models.definitions.ling_3_flash import (
        load_ling_3_flash_config,
        patch_ling_3_flash_remote_code,
        refresh_ling_rotary_buffers,
    )

    patch_ling_3_flash_remote_code()
    config = load_ling_3_flash_config(model_dir)
    config._attn_implementation = "eager"
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=trust_remote_code,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
        output_loading_info=True,
    )
    invalid = {
        key: loading_info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if invalid:
        raise RuntimeError(f"Ling source checkpoint did not load completely: {invalid}")
    refresh_ling_rotary_buffers(model)
    from gptqmodel.models.definitions.ling3_flash_mla import (
        XH2_MLA_LAYOUT, absorb_ling_mla, finalize_ling_mla_checkpoint, validate_mla_layout,
    )

    validate_mla_layout(mla_weight_layout)
    if mla_weight_layout == XH2_MLA_LAYOUT:
        absorb_ling_mla(model)
    patch_ling_3_flash_remote_code()
    layer_config = build_autoround_layer_config(
        model,
        expert_bits=expert_bits,
        group_size=group_size,
    )
    block_names = build_autoround_block_names(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    calibration_source = _calibration_loader(
        calibration_source,
        tokenizer,
        seqlen=seqlen,
        nsamples=nsamples,
        batch_size=batch_size,
        seed=seed,
    )
    autoround = AutoRound(
        model=model,
        tokenizer=tokenizer,
        bits=bits,
        group_size=group_size,
        sym=True,
        data_type=data_type,
        dataset=calibration_source,
        nsamples=nsamples,
        seqlen=seqlen,
        iters=iters,
        batch_size=batch_size,
        gradient_accumulate_steps=1,
        low_gpu_mem_usage=True,
        low_cpu_mem_usage=True,
        device_map=_autoround_device_map(device_map),
        seed=seed,
        layer_config=layer_config,
        to_quant_block_names=block_names,
        enable_quanted_input=True,
    )
    _install_autoround_nonfinite_loss_guard(autoround)
    autoround.quantize()
    patch_ling_3_flash_remote_code()
    autoround.save_quantized(output_dir, format="auto_round:gptqmodel")
    canonicalize_autoround_gptqmodel_config(output_dir)
    if mla_weight_layout == XH2_MLA_LAYOUT:
        finalize_ling_mla_checkpoint(output_dir)
