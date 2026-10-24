"""Qwen3.5 MoE static block precision search, independent of the final solver.

The screening surrogate is symmetric group RTN, NOT calibrated GPTQ/AutoRound.
Output KL gates the cumulative assignment; reconstruction error only orders
trials. Non-target weights and native routing are unchanged during screening.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


REPORT_NAME = "auto_mixed_precision.json"
TARGET = re.compile(
    r"^(?P<prefix>.*\blayers\.(?P<block>\d+)\.)"
    r"(?P<role>self_attn|linear_attn|mlp\.shared_expert)\."
    r"(?P<projection>q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
)
PROJECTIONS = {
    "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "linear_attn": {"in_proj_qkv", "in_proj_z", "out_proj"},
    "mlp.shared_expert": {"gate_proj", "up_proj", "down_proj"},
}


@dataclass(frozen=True)
class AutoMixedPrecisionConfig:
    candidates: tuple[int, ...] = (4, 5, 8)
    initial_bits: int = 5
    max_kl: float = 0.03
    max_ppl_increase: float = 0.02
    nsamples: int = 32
    seqlen: int = 512
    positions_per_sample: int = 64
    max_trials: int = 24
    seed: int = 42
    dataset: str | None = None
    text_key: str = "text"
    device: str | None = None
    execution_mode: str = "layerwise"
    image_dataset: str | None = None
    image_nsamples: int = 32
    image_seqlen: int = 2048
    image_max_pixels: int = 393216


def parse_auto_mixed_precision(value):
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("auto_mixed_precision must be a boolean or mapping")
    unknown = set(value) - set(AutoMixedPrecisionConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown auto_mixed_precision fields: {sorted(unknown)}")
    config = AutoMixedPrecisionConfig(**value)
    if config.execution_mode not in ("layerwise", "resident"):
        raise ValueError("auto_mixed_precision.execution_mode must be layerwise or resident")
    candidates = config.candidates
    if (not isinstance(candidates, (list, tuple)) or not candidates
            or any(type(b) is not int or b not in (4, 5, 8) for b in candidates)
            or len(set(candidates)) != len(candidates) or 8 not in candidates):
        raise ValueError("candidates must contain distinct W4/W5/W8 values including W8")
    if type(config.initial_bits) is not int or config.initial_bits not in candidates:
        raise ValueError("initial_bits must be one of candidates")
    for key in ("nsamples", "seqlen", "positions_per_sample", "max_trials",
                "image_nsamples", "image_seqlen", "image_max_pixels"):
        if type(getattr(config, key)) is not int or getattr(config, key) < 1:
            raise ValueError(f"auto_mixed_precision.{key} must be a positive integer")
    if config.seqlen < 2 or config.positions_per_sample >= config.seqlen:
        raise ValueError("positions_per_sample must be smaller than seqlen (next-token scoring)")
    for key in ("max_kl", "max_ppl_increase"):
        v = getattr(config, key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"auto_mixed_precision.{key} must be finite and nonnegative")
    if type(config.seed) is not int:
        raise ValueError("auto_mixed_precision.seed must be an integer")
    for key in ("dataset", "device", "image_dataset"):
        if getattr(config, key) is not None and not isinstance(getattr(config, key), str):
            raise ValueError(f"auto_mixed_precision.{key} must be a string")
    if config.image_dataset == "" or config.image_seqlen < 2:
        raise ValueError("image_dataset must be nonempty and image_seqlen must be at least 2")
    if not isinstance(config.text_key, str) or not config.text_key:
        raise ValueError("auto_mixed_precision.text_key must be a nonempty string")
    return config


def recipe_search_options(value, *, topology, rotation=False, incompatible=False):
    config = parse_auto_mixed_precision(value)
    if config is None:
        return None
    if topology != "moe" or rotation or incompatible:
        raise ValueError("auto_mixed_precision requires Qwen3.5 MoE, no rotation, and full native-float quantization")
    return asdict(config)


def validate_search_paths(model_dir, output_dir):
    source, target = Path(model_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if source == target:
        raise ValueError("Automatic precision search must not overwrite the source checkpoint")
    if (target / "config.json").exists():
        raise ValueError("Automatic precision output already contains a model; choose a new directory")
    config = source / "config.json"
    if config.is_file() and json.loads(config.read_text()).get("quantization_config"):
        raise ValueError("Automatic precision search requires native floating input, not a packed checkpoint")


def discover_units(model):
    config = getattr(model.config, "text_config", model.config)
    if config.model_type not in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        raise ValueError("auto_mixed_precision currently supports native Qwen3.5/Qwen3.6 MoE only")
    if getattr(model.config, "quantization_config", None):
        raise ValueError("Search requires native floating weights; packed checkpoints cannot be searched again")
    units = {}
    for name, module in model.named_modules():
        match = TARGET.fullmatch(name)
        if not match:
            continue
        role = match["role"]
        if match["projection"] not in PROJECTIONS[role]:
            continue
        if not isinstance(module, torch.nn.Linear) or hasattr(module, "qweight"):
            raise ValueError(f"Search target is not a floating Linear: {name}")
        unit = f"{int(match['block'])}:{'shared_expert' if role == 'mlp.shared_expert' else 'attention'}"
        units.setdefault(unit, {})[name] = module
    expected = {f"{i}:{role}" for i in range(config.num_hidden_layers) for role in ("attention", "shared_expert")}
    if set(units) != expected:
        raise ValueError("Incomplete Qwen MoE attention/shared-expert block coverage")
    for modules in units.values():
        names = list(modules)
        role = TARGET.fullmatch(names[0])["role"]
        if {n.rsplit(".", 1)[1] for n in names} != PROJECTIONS[role]:
            raise ValueError(f"Incomplete search group: {names}")
    return units


def symmetric_qdq(weight, bits, group_size):
    """Signed group RTN screening surrogate with positive FP32 scales."""
    if weight.ndim != 2 or group_size < 1 or weight.shape[1] % group_size:
        raise ValueError("Search requires complete positive-size input groups")
    w = weight.float().reshape(weight.shape[0], -1, group_size)
    qmax = 2 ** (bits - 1) - 1
    scale = (w.abs().amax(-1, keepdim=True) / qmax).clamp_min(torch.finfo(torch.float32).tiny)
    return (torch.round(w / scale).clamp(-qmax - 1, qmax) * scale).reshape_as(weight).to(weight.dtype)


def output_metrics(reference, candidate, labels):
    """Full-vocabulary KL(P_reference || Q_candidate), nats per scored token."""
    if reference.shape != candidate.shape or reference.ndim != 2 or labels.shape != reference.shape[:1]:
        raise ValueError("Expected matching [tokens, vocab] log probabilities and [tokens] labels")
    if not len(labels) or not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Search metrics require nonempty finite log probabilities")
    kl = (reference.exp() * (reference - candidate)).sum(-1).clamp_min(0)
    rows = torch.arange(len(labels), device=labels.device)
    delta = (reference[rows, labels] - candidate[rows, labels]).mean().item()
    return {"mean_kl": kl.mean().item(), "p99_kl": torch.quantile(kl, 0.99).item(),
            "delta_nll": delta, "ppl_increase": math.expm1(min(delta, 80)), "tokens": len(labels)}


def metrics_pass(metrics, config):
    """Each enabled modality must independently satisfy both output budgets."""
    groups = metrics.get("modalities", {"text": metrics})
    required = {"text", "image"} if config.image_dataset else {"text"}
    if not required.issubset(groups):
        return False
    return all(math.isfinite(m["mean_kl"]) and math.isfinite(m["ppl_increase"])
               and m["mean_kl"] <= config.max_kl
               and m["ppl_increase"] <= config.max_ppl_increase for m in groups.values())


def select_assignment(units, errors, evaluate, config):
    """Bounded greedy search; every acceptance is measured on the whole model."""
    plan = {u: config.initial_bits for u in units}
    trials = []
    measured = {}

    def measure(proposal, change):
        key = tuple(sorted(proposal.items()))
        if key in measured:
            return measured[key][0]
        metrics = evaluate(proposal)
        passed = metrics_pass(metrics, config)
        trials.append({"change": change, "unit_bits": dict(proposal), "metrics": metrics, "accepted": passed})
        measured[key] = (passed, metrics)
        print(f"[auto_mixed_precision] trial={len(trials)} {change} KL={metrics['mean_kl']:.6g} "
              f"PPL_delta={metrics['ppl_increase']:.4%} accepted={passed}", flush=True)
        if "modalities" in metrics:
            print(f"[auto_mixed_precision] modalities={json.dumps(metrics['modalities'])}", flush=True)
        return passed

    accepted = measure(plan, "initial")
    # Reach the all-W8 fallback within the configured budget if necessary.
    # On large models one-at-a-time promotions can exhaust every trial without
    # ever evaluating a feasible plan. Batch promotions, then refine downward.
    ordered = sorted(units, key=lambda u: (-(errors[u][config.initial_bits] - errors[u][8]), u))
    batch_size = max(1, math.ceil(len(ordered) / max(1, config.max_trials - 1)))
    for start in range(0, len(ordered), batch_size):
        if accepted or len(trials) >= config.max_trials:
            break
        promoted = [u for u in ordered[start:start + batch_size] if plan[u] != 8]
        if not promoted:
            continue
        for unit in promoted:
            plan[unit] = 8
        accepted = measure(plan, f"{','.join(promoted)}->8")
    if not accepted:
        return {"status": "no_feasible_assignment", "trials": trials, "unit_bits": None}
    # Remove unnecessary W8 promotions in batches. Bisect only rejected
    # batches, so a few sensitive groups do not strand dozens of safe groups
    # at W8 merely because the budget cannot afford testing each one.
    promoted = [u for u in ordered if plan[u] == 8 and config.initial_bits < 8]
    pending = [promoted] if promoted else []
    while pending and len(trials) < config.max_trials:
        group = pending.pop(0)
        proposal = {**plan, **{u: config.initial_bits for u in group}}
        if measure(proposal, f"restore {','.join(group)}->{config.initial_bits}"):
            plan = proposal
        elif len(group) > 1:
            middle = len(group) // 2
            pending.extend((group[:middle], group[middle:]))
    # Spend the remaining budget on the cheapest predicted loss per logical bit saved.
    blocked = set()
    while len(trials) < config.max_trials:
        options = []
        for unit, bits in plan.items():
            lower = [b for b in config.candidates if b < bits]
            if lower:
                target = max(lower)
                if (unit, target) not in blocked:
                    cost = (errors[unit][target] - errors[unit][bits]) / (units[unit] * (bits - target))
                    options.append((cost, unit, target))
        if not options:
            break
        _, unit, target = min(options)
        proposal = dict(plan, **{unit: target})
        if measure(proposal, f"{unit}->{target}"):
            plan = proposal
        else:
            blocked.add((unit, target))
    return {"status": "accepted", "trials": trials, "unit_bits": plan,
            "selected_metrics": measured[tuple(sorted(plan.items()))][1],
            "budget_exhausted": len(trials) == config.max_trials}


def _records(samples, config):
    records = []
    for sample in samples:
        if any(key in sample for key in ("pixel_values", "pixel_values_videos", "images", "videos")):
            raise ValueError("Expected text samples only in the text pool; use image_dataset for images")
        ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
            raise ValueError("Each search record must contain one token sequence")
        ids = ids.reshape(-1)[:config.seqlen].cpu()
        mask = torch.as_tensor(sample.get("attention_mask", torch.ones_like(ids))).reshape(-1)[:len(ids)].cpu()
        if len(mask) != len(ids) or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Invalid attention mask in search data")
        valid = ((mask[:-1] == 1) & (mask[1:] == 1)).nonzero().flatten()
        if not len(valid):
            continue
        positions = valid[torch.linspace(0, len(valid) - 1, min(len(valid), config.positions_per_sample)).long()]
        records.append((ids.unsqueeze(0), mask.unsqueeze(0), positions, ids[positions + 1]))
        if len(records) == config.nsamples:
            return records
    raise ValueError(f"Search needs {config.nsamples} usable text samples, found {len(records)}")


def _load_search_image(row, manifest_dir):
    """Decode one local or inline image, checking its original encoded bytes."""
    from PIL import Image, UnidentifiedImageError

    sources = [key for key in ("image", "image_base64") if key in row]
    if len(sources) != 1:
        raise ValueError("Image JSONL requires exactly one of image or image_base64")
    key = sources[0]
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Image JSONL requires a nonempty {key} string")
    if key == "image_base64":
        try:
            payload = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 must contain valid raw Base64") from exc
    else:
        image_path = Path(value).expanduser()
        if not image_path.is_absolute():
            image_path = Path(manifest_dir) / image_path
        payload = image_path.read_bytes()
    if "image_sha256" in row:
        expected = row["image_sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError("image_sha256 must be a 64-character hexadecimal digest")
        if hashlib.sha256(payload).hexdigest() != expected.lower():
            raise ValueError("Image SHA256 mismatch")
    try:
        with Image.open(io.BytesIO(payload)) as source_image:
            return source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image payload cannot be decoded") from exc


def load_image_records(model_source, config):
    """Local JSONL: image path or Base64, prompt, answer; optional option labels."""
    from transformers import AutoProcessor

    path = Path(config.image_dataset).expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < config.image_nsamples:
        raise ValueError(f"Image search needs {config.image_nsamples} records, found {len(rows)}")
    random.Random(config.seed).shuffle(rows)
    processor = AutoProcessor.from_pretrained(model_source)
    size = processor.image_processor.size
    processor.image_processor.size = {
        "shortest_edge": min(size["shortest_edge"], config.image_max_pixels),
        "longest_edge": config.image_max_pixels,
    }
    records = []
    for row in rows[:config.image_nsamples]:
        if not isinstance(row, Mapping) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                               for k in ("text", "answer")):
            raise ValueError("Image JSONL requires nonempty text and answer strings")
        picture = _load_search_image(row, path.parent)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": picture}, {"type": "text", "text": row["text"]}]}]
        inputs = dict(processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt"))
        if not {"pixel_values", "image_grid_thw", "mm_token_type_ids"}.issubset(inputs):
            raise ValueError("Image processor must produce pixels, grid and multimodal token types")
        answer = torch.tensor(processor.tokenizer.encode(row["answer"], add_special_tokens=False), dtype=torch.long)
        prompt_len = inputs["input_ids"].shape[-1]
        if not len(answer) or prompt_len + len(answer) > config.image_seqlen:
            raise ValueError("Image prompt/answer exceeds image_seqlen or has no answer tokens; do not truncate images")
        inputs["input_ids"] = torch.cat((inputs["input_ids"], answer.unsqueeze(0)), -1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(answer).unsqueeze(0)), -1)
        inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(answer).unsqueeze(0)), -1)
        answer_positions = torch.linspace(0, len(answer) - 1, min(len(answer), config.positions_per_sample)).long()
        options = row.get("options", [])
        if not isinstance(options, (list, dict)) or any(not isinstance(label, str) for label in options):
            raise ValueError("Image options must be a list or mapping of answer labels")
        option_ids = {label: processor.tokenizer.encode(label, add_special_tokens=False) for label in options}
        if options and (row["answer"] not in options or any(len(ids) != 1 for ids in option_ids.values())):
            raise ValueError("Choice diagnostics require single-token option labels and a matching answer")
        records.append({"inputs": inputs, "positions": prompt_len - 1 + answer_positions,
                        "labels": answer[answer_positions], "modality": "image",
                        "id": str(row.get("id", len(records))), "answer": row["answer"],
                        "options": {label: ids[0] for label, ids in option_ids.items()}})
    return records


def aggregate_metrics(items):
    tokens = sum(m["tokens"] for m in items)
    delta = sum(m["delta_nll"] * m["tokens"] for m in items) / tokens
    return {"mean_kl": sum(m["mean_kl"] * m["tokens"] for m in items) / tokens,
            "max_sample_p99_kl": max(m["p99_kl"] for m in items), "delta_nll": delta,
            "ppl_increase": math.expm1(min(delta, 80)), "tokens": tokens, "samples": len(items)}


@torch.inference_mode()
def search_precision(model, samples, value, *, group_size=64, device="cuda:0", image_samples=None):
    config = parse_auto_mixed_precision(value)
    if config is None:
        raise ValueError("search_precision requires enabled auto_mixed_precision")
    units = discover_units(model)
    modules = {n: m for group in units.values() for n, m in group.items()}
    original_device = {t.device for t in list(model.parameters()) + list(model.buffers())}
    if len(original_device) != 1 or any(hasattr(m, "_hf_hook") for m in model.modules()):
        raise ValueError("Search requires an undispatched model on one device; disable device_map/disk offload")
    original_device = original_device.pop()
    if original_device.type == "meta":
        raise ValueError("Search cannot read meta/offloaded weights")
    records = [{"inputs": {"input_ids": ids, "attention_mask": mask}, "positions": positions,
                "labels": labels, "modality": "text"} for ids, mask, positions, labels in _records(samples, config)]
    if config.image_dataset:
        if image_samples is None or len(image_samples) != config.image_nsamples:
            raise ValueError("image_dataset requires the configured number of processed image samples")
        if not hasattr(model.model, "visual") or not hasattr(model.model, "language_model"):
            raise ValueError("Image search requires the native multimodal model with a vision tower")
        records.extend({k: v for k, v in r.items() if k != "cached_inputs"} for r in image_samples)
    elif image_samples:
        raise ValueError("Set image_dataset when supplying image samples")
    device = torch.device(config.device or device)
    originals = {n: m.weight.detach().cpu().clone() for n, m in modules.items()}
    training = {m: m.training for m in model.modules()}
    rope_owner = model.model
    original_rope_deltas = getattr(rope_owner, "rope_deltas", None)
    input_cache, hooks = {}, []
    candidates, errors = {}, {u: {} for u in units}
    digest = hashlib.sha256()
    weight_digest = hashlib.sha256()
    for name, weight in originals.items():
        weight_digest.update(name.encode())
        weight_digest.update(weight.view(torch.uint8).numpy().tobytes())
    for record in records:
        digest.update(record["modality"].encode())
        for key, tensor in sorted({**record["inputs"], "positions": record["positions"],
                                   "labels": record["labels"]}.items()):
            digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    try:
        print(f"[auto_mixed_precision] execution_mode={config.execution_mode} device={device}", flush=True)
        model.eval().to("cpu" if config.execution_mode == "layerwise" else device)
        replay = None
        if config.execution_mode == "layerwise":
            from .auto_mixed_precision_replay import LayerwiseSearchReplay
            replay = LayerwiseSearchReplay(model, device)
            replay.prepare(records)
        # A bounded set of real Linear inputs ranks candidate reconstruction loss.
        def capture(name):
            def hook(module, args):
                x = args[0].detach().reshape(-1, module.in_features)
                positions = torch.linspace(0, len(x) - 1, min(len(x), 16), device=x.device).long()
                input_cache.setdefault(name, []).append(x[positions].cpu())
            return hook
        for name, module in modules.items():
            hooks.append(module.register_forward_pre_hook(capture(name)))

        def forward(record, capture_vision=False):
            # Vision and embeddings never change in this search. Cache the actual
            # native language-model input and M-RoPE positions during the reference
            # pass, then replay them through the complete conditional model.
            vision_hook = None
            def capture_language_input(module, args, kwargs):
                record["cached_inputs"] = {k: kwargs[k].detach().cpu().clone()
                                           for k in ("inputs_embeds", "position_ids", "attention_mask")}
            if capture_vision and record["modality"] == "image":
                vision_hook = model.model.language_model.register_forward_pre_hook(
                    capture_language_input, with_kwargs=True)
            try:
                inputs = record.get("cached_inputs", record["inputs"])
                out = model(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False,
                            logits_to_keep=record["positions"].to(device), return_dict=True)
            finally:
                if vision_hook is not None:
                    vision_hook.remove()
            return F.log_softmax(out.logits[0].float(), dim=-1).cpu()

        reference = []
        if replay is not None:
            reference = replay.forward(records)
            print(f"[auto_mixed_precision] layerwise reference {len(records)}/{len(records)}", flush=True)
        else:
            for i, record in enumerate(records):
                reference.append(forward(record, capture_vision=True))
                if (i + 1) % 8 == 0 or i + 1 == len(records):
                    print(f"[auto_mixed_precision] reference {i + 1}/{len(records)}", flush=True)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        for unit, group in units.items():
            for bits in config.candidates:
                error = 0.0
                for name, module in group.items():
                    w = originals[name].to(device)
                    q = symmetric_qdq(w, bits, group_size)
                    candidates[name, bits] = q.cpu()
                    x = torch.cat(input_cache[name]).to(device).float()
                    y = F.linear(x, w.float())
                    error += ((F.linear(x, q.float()) - y).square().mean()
                              / y.square().mean().clamp_min(1e-12)).item()
                    del w, q, x, y
                errors[unit][bits] = error / len(group)
        input_cache.clear()
        counts = {u: sum(m.weight.numel() for m in group.values()) for u, group in units.items()}

        def evaluate(plan, force_all=False):
            for unit, group in units.items():
                for name, module in group.items():
                    module.weight.copy_(candidates[name, plan[unit]])
            by_modality = {}
            for modality in ("text", "image") if config.image_dataset else ("text",):
                items, choices = [], []
                pairs = [(ref, record) for ref, record in zip(reference, records)
                         if record["modality"] == modality]
                predictions = (replay.forward([record for _, record in pairs]) if replay is not None
                               else (forward(record) for _, record in pairs))
                for (ref, record), candidate in zip(pairs, predictions):
                    items.append(output_metrics(ref, candidate, record["labels"]))
                    if record.get("options"):
                        options = record["options"]
                        teacher_choice = max(options, key=lambda label: ref[0, options[label]].item())
                        choice = max(options, key=lambda label: candidate[0, options[label]].item())
                        choices.append({"id": record["id"], "reference": teacher_choice, "candidate": choice,
                                        "answer": record["answer"]})
                by_modality[modality] = aggregate_metrics(items)
                if choices:
                    by_modality[modality]["choices"] = {
                        "count": len(choices), "reference_correct": sum(c["reference"] == c["answer"] for c in choices),
                        "candidate_correct": sum(c["candidate"] == c["answer"] for c in choices),
                        "agreement": sum(c["reference"] == c["candidate"] for c in choices) / len(choices),
                        "details": choices}
                # A rejected text proposal cannot be accepted by image scores.
                # Skip its image replay, while recording that images were unmeasured.
                text = by_modality[modality]
                if not force_all and modality == "text" and (text["mean_kl"] > config.max_kl
                                                             or text["ppl_increase"] > config.max_ppl_increase):
                    break
            if not config.image_dataset:
                return by_modality["text"]
            result = {k: max(m[k] for m in by_modality.values())
                      for k in ("mean_kl", "max_sample_p99_kl", "delta_nll", "ppl_increase")}
            result.update(tokens=sum(m["tokens"] for m in by_modality.values()), modalities=by_modality,
                          skipped_modalities=sorted({"text", "image"} - set(by_modality)))
            return result

        all_w8_metrics = evaluate({u: 8 for u in units}, force_all=True)
        print(f"[auto_mixed_precision] all-W8 control KL={all_w8_metrics['mean_kl']:.6g} "
              f"PPL_delta={all_w8_metrics['ppl_increase']:.4%}", flush=True)
        report = select_assignment(counts, errors, evaluate, config)
        report.update({"version": 1, "config": asdict(config), "group_size": group_size,
                       "surrogate": "symmetric_group_rtn", "reference": "native_float",
                       "metric": ("full_vocab_forward_kl_nats_T1_text_and_image_answer"
                                  if config.image_dataset else "full_vocab_forward_kl_nats_T1_text_next_token"),
                       "data_sha256": digest.hexdigest(), "sample_count": len(records),
                       "modality_sample_counts": {m: sum(r["modality"] == m for r in records)
                                                  for m in {r["modality"] for r in records}},
                       "all_w8_metrics": all_w8_metrics, "control_evaluations": 1,
                       "target_weight_sha256": weight_digest.hexdigest(),
                       "model_config": model.config.to_dict(), "unit_numel": counts,
                       "reconstruction_errors": errors, "final_quantization_validated": False})
        if report["unit_bits"] is not None:
            report["module_bits"] = {n: report["unit_bits"][u] for u, group in units.items() for n in group}
            report["logical_weight_bytes"] = sum(counts[u] * b / 8 for u, b in report["unit_bits"].items())
            report["logical_bytes_saved_vs_w8"] = sum(counts.values()) - report["logical_weight_bytes"]
        return report
    finally:
        for hook in hooks:
            hook.remove()
        for name, module in modules.items():
            module.weight.copy_(originals[name])
        model.to(original_device)
        if hasattr(rope_owner, "rope_deltas"):
            rope_owner.rope_deltas = original_rope_deltas
        for module, was_training in training.items():
            module.training = was_training


def run_search(model, *, tokenizer, value, dataset, output_dir, group_size, device):
    """CLI bridge: resolve deterministic text/image data and persist search evidence."""
    config = parse_auto_mixed_precision(value)
    if config is None:
        return None
    from ..recipes.qwen35_head import _load_calibration
    source = config.dataset or dataset
    if not isinstance(source, str) or not source:
        raise ValueError("Automatic precision search requires a text dataset or JSONL path")
    local = Path(source).expanduser().is_file()
    samples = _load_calibration(
        tokenizer.name_or_path, Path(source).expanduser() if local else None,
        None if local else source, config.text_key, config.nsamples, config.seqlen, config.seed)
    images = load_image_records(tokenizer.name_or_path, config) if config.image_dataset else None
    report = search_precision(model, samples, asdict(config), group_size=group_size, device=device,
                              image_samples=images)
    report["dataset"] = str(source)
    report["model_source"] = str(tokenizer.name_or_path)
    report_path = Path(str(output_dir) + ".auto_mixed_precision_search.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "accepted":
        raise ValueError(f"No feasible mixed precision assignment within the search budget; report: {report_path}")
    return report


def dynamic_overrides(report, existing=None):
    # GPTQ dynamic_get is first-match; exact searched overrides must precede W8 regexes.
    return {**{f"+:^{re.escape(n)}$": {"bits": b} for n, b in report["module_bits"].items()},
            **(existing or {})}


def save_search_report(output_dir, report):
    """Attach the logical assignment to both standard quant configs and a report."""
    if report is None:
        return
    root = Path(output_dir)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    quant = config["quantization_config"]
    from .config import dynamic_get
    for name, bits in report["module_bits"].items():
        actual = dynamic_get(quant.get("dynamic"), name, "bits", quant.get("bits"))
        extra = quant.get("extra_config", {}).get(name, {})
        actual = extra.get("bits", actual)
        if actual != bits:
            raise ValueError(f"Saved precision disagrees with search: {name}: {actual} != {bits}")
    plan = {k: report[k] for k in ("version", "status", "group_size", "module_bits", "unit_bits")}
    plan["report"] = REPORT_NAME
    quant.setdefault("meta", {})["auto_mixed_precision"] = plan
    _validate_precision_plan(root, config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for name in ("quantize_config.json", "quant_config.json"):
        path = root / name
        if path.exists():
            q = json.loads(path.read_text())
            q.setdefault("meta", {})["auto_mixed_precision"] = plan
            path.write_text(json.dumps(q, indent=2) + "\n")
    (root / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")


def read_precision_plan(model_dir):
    """Validate logical metadata against GPTQ configuration and packed shapes."""
    root = Path(model_dir)
    return _validate_precision_plan(root, json.loads((root / "config.json").read_text()))


def _validate_precision_plan(root, config):
    from safetensors import safe_open
    from .config import dynamic_get

    quant = config.get("quantization_config", {})
    plan = quant.get("meta", {}).get("auto_mixed_precision")
    if not isinstance(plan, dict) or plan.get("version") != 1 or plan.get("status") != "accepted":
        raise ValueError("Missing or unsupported accepted automatic precision plan")
    modules = plan.get("module_bits")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Empty automatic precision plan")
    config_text = config.get("text_config", config)
    group_size = plan.get("group_size")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("Invalid group size in automatic precision plan")
    expected_units = {f"{i}:{role}" for i in range(config_text["num_hidden_layers"])
                      for role in ("attention", "shared_expert")}
    actual_groups = {}
    shapes = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.rsplit(".", 1)[0] in modules:
                    shapes[key] = tensors.get_slice(key).get_shape()
    for name, bits in modules.items():
        match = TARGET.fullmatch(name)
        if (match is None or match["projection"] not in PROJECTIONS[match["role"]]
                or type(bits) is not int or bits not in (4, 5, 8)):
            raise ValueError(f"Unsupported automatic precision entry: {name}: {bits}")
        role = "shared_expert" if match["role"] == "mlp.shared_expert" else "attention"
        unit = f"{int(match['block'])}:{role}"
        actual_groups.setdefault(unit, set()).add(match["projection"])
        if plan.get("unit_bits", {}).get(unit) != bits:
            raise ValueError(f"Per-module precision disagrees with block group {unit}")
        actual = dynamic_get(quant.get("dynamic"), name, "bits", quant.get("bits"))
        actual = quant.get("extra_config", {}).get(name, {}).get("bits", actual)
        if actual != bits:
            raise ValueError(f"Checkpoint configuration conflicts with plan at {name}: {actual} != {bits}")
        actual_group = dynamic_get(quant.get("dynamic"), name, "group_size", quant.get("group_size"))
        actual_group = quant.get("extra_config", {}).get(name, {}).get("group_size", actual_group)
        if actual_group != group_size:
            raise ValueError(f"Checkpoint group size conflicts with plan at {name}")
        qweight = shapes.get(name + ".qweight")
        scales = shapes.get(name + ".scales")
        g_idx = shapes.get(name + ".g_idx")
        qzeros = shapes.get(name + ".qzeros")
        storage_bits = 8 if bits == 5 else bits
        if (not qweight or not scales or not g_idx or not qzeros
                or len(qweight) != 2 or len(scales) != 2 or len(g_idx) != 1 or len(qzeros) != 2
                or qweight[0] * (32 // storage_bits) != g_idx[0]
                or scales[0] * group_size != g_idx[0]
                or qweight[1] != scales[1] or qzeros[0] != scales[0]
                or qzeros[1] * (32 // storage_bits) != scales[1]):
            raise ValueError(f"Packed tensors disagree with W{bits} plan at {name}")
    if set(actual_groups) != expected_units or set(plan.get("unit_bits", {})) != expected_units:
        raise ValueError("Automatic precision plan has incomplete block coverage")
    for unit, projections in actual_groups.items():
        expected = (PROJECTIONS["mlp.shared_expert"] if unit.endswith("shared_expert") else
                    PROJECTIONS["self_attn"] if "q_proj" in projections else PROJECTIONS["linear_attn"])
        if projections != expected:
            raise ValueError(f"Automatic precision plan has incomplete projections in {unit}")
    return plan
