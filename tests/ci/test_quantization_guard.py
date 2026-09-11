"""Small real-quantization guard for APIs consumed by xh2modelzoo.

Uses a floating ModelZoo block, reduced to representative linear/full attention
layers, for GPTQ and AutoRound W4/W8 smoke tests. Other structural mechanisms
need their own block tests. Opt-in keeps upstream test enumeration off the GPU.
"""
from __future__ import annotations

import functools
import json
import os
import tempfile
from pathlib import Path

import pytest

from _fixture_download import _download, _safe_extract


RUN_QUANT_GUARD = os.environ.get("GPTQMODEL_RUN_QUANT_GUARD") == "1"


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_URL = (
    "http://10.10.1.53:8081/artifactory/model_zoo2/"
    "ci/gptqmodel-blocks-v1/qwen35-dense-min2-fp-v1.tar.gz"
)
# Pinned at publication and checked against Artifactory's stored checksum.
DEFAULT_ARCHIVE_SHA256 = "1027dc6b33d4f665b16db4ce57b6fd862a9f9490a49464cd23d88d4d73ea6d71"
DEFAULT_CACHE_ROOT = REPO_ROOT / "work_dirs/ci-block-fixtures/guard-cache"


@functools.lru_cache(maxsize=1)
def _model_dir() -> Path:
    explicit = os.environ.get("GPTQMODEL_QUANT_GUARD_MODEL_DIR")
    if explicit:
        model_dir = Path(explicit).expanduser().resolve()
    else:
        url = os.environ.get("GPTQMODEL_QUANT_GUARD_ARCHIVE_URL") or DEFAULT_ARCHIVE_URL
        expected = (os.environ.get("GPTQMODEL_QUANT_GUARD_SHA256") or
                    (DEFAULT_ARCHIVE_SHA256 if url == DEFAULT_ARCHIVE_URL else "")).lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(
                "Set GPTQMODEL_QUANT_GUARD_MODEL_DIR to a prepared checkpoint, or provide "
                "GPTQMODEL_QUANT_GUARD_SHA256 for the downloaded archive"
            )
        cache_root = Path(os.environ.get("GPTQMODEL_QUANT_GUARD_CACHE", DEFAULT_CACHE_ROOT)).expanduser()
        archive = cache_root / f"{expected}.tar.gz"
        _download(url, archive, expected)
        # Each run extracts verified bytes into its own directory. Avoid stale
        # or partially extracted shared model caches between GPU workers.
        work = tempfile.TemporaryDirectory(dir=cache_root, prefix="model-")
        _MODEL_WORK_DIRS.append(work)
        model_dir = _safe_extract(archive, Path(work.name))
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"trimmed quant guard model has no config.json: {model_dir}")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("this guard requires a Qwen3.5 dense floating fixture")
    from prepare_block import prepare

    work = tempfile.TemporaryDirectory(prefix="qwen35-ci-block-")
    _MODEL_WORK_DIRS.append(work)
    block = Path(work.name) / "checkpoint"
    report = prepare(model_dir, block)
    print("CI block fixture: " + json.dumps(report), flush=True)
    return block


_MODEL_WORK_DIRS: list[tempfile.TemporaryDirectory] = []


def _assert_scales(scales, source_weight, group_size):
    import torch

    assert torch.isfinite(scales).all() and (scales >= 0).all()
    assert (scales > 0).any(), "an entire quantized module cannot have zero scales"
    if (scales == 0).any():
        # The real Qwen checkpoint contains ~1e-37 Q/K channels. They round to
        # zero in FP16 storage. Accept zero scales only when that whole source
        # group itself is zero at the stored dtype; never hide lost normal weights.
        maximum = source_weight.float().reshape(source_weight.shape[0], -1, group_size).abs().amax(-1).T
        assert maximum.shape == scales.shape
        assert (maximum.to(scales.dtype)[scales == 0] == 0).all(), "zero scale erased a representable source group"


def _assert_quant_artifacts(output_dir: Path, provenance_name: str, bits: int, source_dir: Path) -> None:
    config_path = output_dir / "config.json"
    assert config_path.is_file()
    assert (output_dir / provenance_name).is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_config = config.get("quantization_config")
    assert isinstance(quant_config, dict), "saved config does not describe a quantized checkpoint"
    assert int(quant_config["bits"]) == bits
    assert int(quant_config["group_size"]) == 64
    weight_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("pytorch_model*.bin"))
    assert weight_files, f"no quantized weight artifacts were produced under {output_dir}"

    from safetensors import safe_open
    import torch

    # A floating checkpoint with quantization metadata must not pass this gate.
    packed = set()
    layers = set()
    for weights in output_dir.glob("*.safetensors"):
        with safe_open(weights, framework="pt", device="cpu") as reader:
            packed.update(key.rsplit(".", 1)[-1] for key in reader.keys())
            for key in reader.keys():
                if key.endswith(".qweight"):
                    assert reader.get_tensor(key).dtype in (torch.int32, torch.int16, torch.int8)
                    if ".layers." in key:
                        layers.add(int(key.split(".layers.", 1)[1].split(".", 1)[0]))
                elif key.endswith(".scales"):
                    scales = reader.get_tensor(key)
                    with safe_open(source_dir / "model.safetensors", framework="pt", device="cpu") as source:
                        original = source.get_tensor(key.removesuffix(".scales") + ".weight")
                    _assert_scales(scales, original, 64)
    assert {"qweight", "qzeros", "scales"} <= packed, "missing packed quantization tensors"
    assert layers == {0, 1}, "both representative attention layers must produce packed weights"
    text_config = config.get("text_config", config)
    assert text_config["layer_types"] == ["linear_attention", "full_attention"]


def _assert_reload(output_dir: Path) -> None:
    import torch

    from gptqmodel import BACKEND, GPTQModel

    model = GPTQModel.load(str(output_dir), device="cuda:0", backend=BACKEND.TORCH)
    tokens = model.tokenizer("Hello", return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.inference_mode():
        logits = model.model(input_ids=tokens, use_cache=False).logits
    assert logits.shape[:2] == tokens.shape
    assert torch.isfinite(logits).all()
    del model
    torch.cuda.empty_cache()


@pytest.mark.quant_guard
@pytest.mark.cuda
@pytest.mark.skipif(
    not RUN_QUANT_GUARD,
    reason="real quantization guard is enabled only by the dedicated GPTQModel workflow",
)
@pytest.mark.parametrize("backend", ("gptq", "autoround"))
@pytest.mark.parametrize("bits", (4, 8))
def test_qwen35_real_quantize_and_save_guard(backend: str, bits: int, tmp_path: Path) -> None:
    import torch

    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    assert torch.cuda.is_available(), "dedicated quantization guard requires a CUDA runner"
    model_dir = _model_dir()
    output_dir = tmp_path / f"qwen35-{backend}"

    if backend == "gptq":
        calibration_jsonl = (
            REPO_ROOT
            / "gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
        )
        result = quantize_qwen35(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            method="gptq",
            preset="llm_only",
            topology="dense",
            bits=bits,
            group_size=64,
            device="cuda:0",
            device_map=None,
            batch_size=1,
            calibration_jsonl=str(calibration_jsonl),
            nsamples=4,
            seqlen=128,
            check_quant_vision_demo=False,
            skip_inference=True,
        )
        provenance_name = "qwen35_quant_recipe_provenance.json"
    else:
        calibration_jsonl = (
            REPO_ROOT
            / "gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
        )
        result = quantize_qwen35_autoround(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            topology="dense",
            mode="llm-only",
            bits=bits,
            group_size=64,
            iters=1,
            nsamples=4,
            seqlen=128,
            batch_size=1,
            dataset=str(calibration_jsonl),
            device="cuda:0",
            format="auto_gptq",
        )
        provenance_name = "qwen35_autoround_recipe_provenance.json"

    assert result.output_dir == str(output_dir.resolve())
    _assert_quant_artifacts(output_dir, provenance_name, bits, model_dir)
    _assert_reload(output_dir)
