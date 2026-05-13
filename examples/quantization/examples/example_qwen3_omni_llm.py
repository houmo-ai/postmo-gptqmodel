from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForTextToWaveform, AutoProcessor, AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


DEFAULT_MODEL = "/data01/datasets/Qwen3-Omni-30B-A3B-Instruct"
CONFIG_BRIDGE_ATTRS = ("bos_token_id", "eos_token_id", "pad_token_id", "vocab_size")
OPTIONAL_TEXT_ASSET_FILES = (
    "chat_template.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize the Qwen3 Omni thinker LLM with a regular text calibration dataset"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Qwen3 Omni model path or HF repo id")
    parser.add_argument("--out", type=str, default=None, help="Output directory for the quantized LLM")
    parser.add_argument(
        "--llm-export-dir",
        type=str,
        default=None,
        help="Optional directory used to also export thinker.model as a plain HF checkpoint",
    )
    parser.add_argument(
        "--keep-llm-export-dir",
        action="store_true",
        help="Keep the exported thinker.model directory after quantization finishes",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--rotation",
        type=str,
        default="",
        help="Rotation mode. Leave empty to disable rotation; full Omni checkpoints do not yet support the Qwen3 MoE rotation path.",
    )
    parser.add_argument("--offload-to-disk", action="store_true")
    parser.add_argument("--calibration-jsonl", type=str, default=None)
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--nsamples", type=int, default=1024)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the Omni checkpoint",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        default=True,
        help="Save the extracted thinker.model with safetensors when supported",
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Save the extracted thinker.model as PyTorch bin files instead of safetensors",
    )
    parser.add_argument("--max-shard-size", type=str, default="10GB")
    return parser.parse_args()


def build_default_export_dir(args: argparse.Namespace) -> Path:
    model_path = Path(args.model).expanduser()
    model_name = model_path.name or "qwen3-omni"
    base_dir = model_path.parent if str(model_path.parent) not in {"", "."} else Path("work_dirs")
    return base_dir / f"{model_name}-thinker-llm"


def build_default_quant_out(args: argparse.Namespace) -> Path:
    model_path = Path(args.model).expanduser()
    model_name = model_path.name or "qwen3-omni"
    base_dir = model_path.parent if str(model_path.parent) not in {"", "."} else Path("work_dirs")
    return base_dir / f"{model_name}-llm-gptq-{args.bits}bit-{args.group_size}g"


def get_wikitext2(nsamples: int, seqlen: int) -> list[str]:
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen
    )

    samples: list[str] = []
    for example in traindata:
        text = example.get("text") if isinstance(example, dict) else None
        if isinstance(text, str) and text.strip():
            samples.append(text)
        if len(samples) >= nsamples:
            break

    return samples


def get_jsonl_texts(path: str, nsamples: int, text_key: str = "text") -> list[str]:
    samples: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")

            samples.append(text)
            if len(samples) >= nsamples:
                break

    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")

    return samples


def build_calibration_dataset(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.calibration_jsonl:
        calibration_dataset = get_jsonl_texts(
            path=args.calibration_jsonl,
            nsamples=args.nsamples,
            text_key=args.calibration_text_key,
        )
        source_name = Path(args.calibration_jsonl).name
        return calibration_dataset, f"jsonl:{source_name}"

    return get_wikitext2(nsamples=args.nsamples, seqlen=args.seqlen), "wikitext2"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def patch_qwen3_omni_config_bridge() -> None:
    import transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe as qwen3_omni_config

    if getattr(qwen3_omni_config, "_gptqmodel_qwen3_omni_cfg_bridge", False):
        return

    def _bridge_property(name: str, text_config_getter):
        def _get(instance):
            value = instance.__dict__.get(name, None)
            if value is not None:
                return value

            text_cfg = text_config_getter(instance)
            return getattr(text_cfg, name, None) if text_cfg is not None else None

        def _set(instance, value):
            instance.__dict__[name] = value
            text_cfg = text_config_getter(instance)
            if text_cfg is not None and hasattr(text_cfg, name):
                setattr(text_cfg, name, value)

        return property(_get, _set)

    root_cfg_cls = qwen3_omni_config.Qwen3OmniMoeConfig
    thinker_cfg_cls = qwen3_omni_config.Qwen3OmniMoeThinkerConfig

    root_text_config = lambda instance: getattr(getattr(instance, "thinker_config", None), "text_config", None)
    thinker_text_config = lambda instance: getattr(instance, "text_config", None)

    for attr in CONFIG_BRIDGE_ATTRS:
        setattr(root_cfg_cls, attr, _bridge_property(attr, root_text_config))
        setattr(thinker_cfg_cls, attr, _bridge_property(attr, thinker_text_config))

    setattr(qwen3_omni_config, "_gptqmodel_qwen3_omni_cfg_bridge", True)


def copy_optional_text_assets(model_id: str, save_dir: Path) -> None:
    model_path = Path(model_id).expanduser()
    if not model_path.is_dir():
        return

    for file_name in OPTIONAL_TEXT_ASSET_FILES:
        src = model_path / file_name
        if src.is_file():
            shutil.copy2(src, save_dir / file_name)


def save_tokenizer_assets(model_id: str, export_dir: Path, trust_remote_code: bool) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise

    tokenizer.save_pretrained(str(export_dir))
    copy_optional_text_assets(model_id, export_dir)


@contextmanager
def tolerate_qwen3_omni_processor_import_error():
    from gptqmodel.models.definitions.qwen3_omni_moe import Qwen3OmniMoeGPTQ

    original_after_model_load = Qwen3OmniMoeGPTQ.after_model_load

    def patched_after_model_load(self, model, load_quantized_model=False):
        try:
            return original_after_model_load(self, model, load_quantized_model=load_quantized_model)
        except ImportError as exc:
            print(f"[warn] Skipping Qwen3 Omni AutoProcessor load: {exc}")
            self.processor = None
            return model

    Qwen3OmniMoeGPTQ.after_model_load = patched_after_model_load
    try:
        yield
    finally:
        Qwen3OmniMoeGPTQ.after_model_load = original_after_model_load


def export_thinker_llm(args: argparse.Namespace, export_dir: Path) -> Path:
    reset_dir(export_dir)

    model = AutoModelForTextToWaveform.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    thinker = getattr(model, "thinker", None)
    if thinker is None:
        raise AttributeError("The loaded Qwen3 Omni model has no thinker submodule.")

    llm_model = getattr(thinker, "model", None)
    if llm_model is None:
        raise AttributeError("The loaded Qwen3 Omni model has no thinker.model LLM submodule.")

    if hasattr(llm_model, "save_pretrained"):
        llm_model.save_pretrained(
            str(export_dir),
            safe_serialization=args.safe_serialization,
            max_shard_size=args.max_shard_size,
        )
    else:
        text_config = getattr(getattr(model.config, "thinker_config", None), "text_config", None)
        if text_config is None:
            raise ValueError("Unable to resolve thinker.text_config for Qwen3 Omni LLM export.")

        text_model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=args.trust_remote_code)
        thinker_prefix = "thinker.model."
        filtered_state_dict = {
            key[len(thinker_prefix):]: value
            for key, value in model.state_dict().items()
            if key.startswith(thinker_prefix)
        }
        missing, unexpected = text_model.load_state_dict(filtered_state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected LLM keys during export: {unexpected[:8]}")
        if missing:
            print(f"[warn] Missing keys when loading thinker.model into text-only model: {missing[:8]}")

        text_model.save_pretrained(
            str(export_dir),
            safe_serialization=args.safe_serialization,
            max_shard_size=args.max_shard_size,
        )

    save_tokenizer_assets(args.model, export_dir, args.trust_remote_code)
    return export_dir


def build_quant_config(args: argparse.Namespace) -> QuantizeConfig:
    rotation = args.rotation.strip() if isinstance(args.rotation, str) else args.rotation
    if rotation in {"", "none", "null", "off"}:
        rotation = None

    return QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        rotation=rotation,
        offload_to_disk=args.offload_to_disk,
    )


def main() -> None:
    args = parse_args()

    export_dir = Path(args.llm_export_dir) if args.llm_export_dir else build_default_export_dir(args)
    quant_out = Path(args.out) if args.out else build_default_quant_out(args)
    quant_out.parent.mkdir(parents=True, exist_ok=True)

    export_requested = args.keep_llm_export_dir or args.llm_export_dir is not None
    cleanup_export_dir = export_requested and not args.keep_llm_export_dir and args.llm_export_dir is None
    exported_model_dir = None

    if export_requested:
        print(f"[1/3] Exporting thinker.model from Omni checkpoint to: {export_dir}")
        exported_model_dir = export_thinker_llm(args, export_dir)
    else:
        print("[1/3] Skipping thinker.model export; GPTQModel will quantize only thinker.model layers from the full Omni checkpoint.")

    print(f"[2/3] Loading Qwen3 Omni checkpoint for GPTQ quantization from: {args.model}")
    patch_qwen3_omni_config_bridge()
    quant_config = build_quant_config(args)
    with tolerate_qwen3_omni_processor_import_error():
        model = GPTQModel.load(args.model, quant_config, trust_remote_code=args.trust_remote_code)

    calibration_dataset, calibration_source = build_calibration_dataset(args)
    print(f"[3/3] Running GPTQ quantization with text calibration dataset ({calibration_source}), samples={len(calibration_dataset)} ...")
    model.quantize(calibration=cast(Any, calibration_dataset), batch_size=args.batch_size)
    model.save(str(quant_out))
    save_tokenizer_assets(args.model, quant_out, args.trust_remote_code)
    print(f"量化完成，Qwen3 Omni 中的 LLM 部分已量化并保存到: {quant_out}")

    if cleanup_export_dir and exported_model_dir is not None:
        shutil.rmtree(exported_model_dir, ignore_errors=True)
        print(f"临时 thinker.model 导出目录已清理: {exported_model_dir}")


if __name__ == "__main__":
    main()