from .base import get_calibration_provider, register_calibration_provider
from .jsonl_loader import load_calibration_jsonl
from .moe_ebss_provider import MoeEbssProvider


def _register_default_providers() -> None:
    try:
        get_calibration_provider("moe_ebss")
    except ValueError:
        register_calibration_provider(MoeEbssProvider())


def resolve_calibration_source(model, tokenizer, quantize_config, calibration):
    cfg = getattr(quantize_config, "calibration_generation", None)
    if cfg is None or not cfg.enabled:
        return calibration

    # keep backward compatibility by default: user-provided calibration wins
    if calibration is not None and not cfg.allow_override:
        return calibration

    _register_default_providers()
    provider = get_calibration_provider(cfg.provider)
    generated = provider.generate(model=model, tokenizer=tokenizer, cfg=cfg)

    if isinstance(generated, dict):
        rows = generated.get("rows")
        if rows is not None:
            return rows
        output_path = generated.get("output_path")
        if output_path:
            return load_calibration_jsonl(output_path)

    if isinstance(generated, str):
        return load_calibration_jsonl(generated)

    raise ValueError("Calibration generation: provider returned unsupported artifact.")
