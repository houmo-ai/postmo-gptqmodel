import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import CalibrationProviderBase


class MoeEbssProvider(CalibrationProviderBase):
    name = "moe_ebss"

    def _resolve_script_name(self, model, cfg) -> str:
        model_patch = getattr(cfg, "model_patch", None)
        if isinstance(model_patch, str) and model_patch.strip():
            patch_key = model_patch.strip().lower()
        else:
            model_type = ""
            if model is not None and getattr(model, "config", None) is not None:
                model_type = str(getattr(model.config, "model_type", "")).lower()
            patch_key = "qwen3_next" if "next" in model_type else "qwen3_moe"

        if patch_key == "qwen3_next":
            return "qwen3_next_gen.py"
        return "qwen3_moe_gen.py"

    def _resolve_model_path(self, model) -> str:
        model_path = ""
        if model is not None and getattr(model, "config", None) is not None:
            model_path = str(getattr(model.config, "_name_or_path", "") or "")
        model_path = model_path.strip()
        if not model_path:
            raise ValueError("Calibration generation: unable to resolve model path from model.config._name_or_path")
        return model_path

    def generate(self, model, tokenizer, cfg) -> str:
        output_path = str(cfg.output_path)
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(output_path) and not bool(getattr(cfg, "overwrite", False)):
            raise ValueError(f"Calibration generation: output already exists: {output_path}")

        model_path = self._resolve_model_path(model)
        script_name = self._resolve_script_name(model, cfg)

        moe_root = Path(__file__).resolve().parents[1] / "moe_ebss"
        script_path = moe_root / script_name
        if not script_path.exists():
            raise ValueError(f"Calibration generation: provider script not found: {script_path}")

        gen_dir = os.path.join(output_dir, ".calib_gen_tmp")
        os.makedirs(gen_dir, exist_ok=True)

        command = [
            sys.executable,
            str(script_path),
            "--model_path",
            model_path,
            "--num_beams",
            str(cfg.num_beams),
            "--seqlen",
            str(cfg.seqlen),
            "--nsamples",
            str(cfg.nsamples),
            "--gen_dir",
            gen_dir,
        ]

        subprocess.run(command, check=True)

        generated_path = os.path.join(gen_dir, f"{Path(model_path).name}.jsonl")
        if not os.path.exists(generated_path):
            raise ValueError(f"Calibration generation: expected jsonl is missing: {generated_path}")

        if os.path.abspath(generated_path) != os.path.abspath(output_path):
            shutil.copyfile(generated_path, output_path)

        return output_path
