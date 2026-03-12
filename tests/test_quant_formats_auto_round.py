import json
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel, __version__
from gptqmodel.quantization import FORMAT, QUANT_CONFIG_FILENAME
from gptqmodel.quantization.config import (
    META_FIELD_QUANTIZER,
    META_QUANTIZER_GPTQMODEL,
    AutoRoundQuantizeConfig,
    METHOD as QUANT_METHOD,
    QuantizeConfig,
)
from gptqmodel.utils.torch import torch_empty_cache


def _build_tiny_llama():
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg).eval()
    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
        },
    ]
    return cfg, model, calibration


def test_auto_round_quantize_config_roundtrips_on_saved_model():
    torch.manual_seed(0)
    cfg, model, calibration = _build_tiny_llama()

    with tempfile.TemporaryDirectory() as native_dir:
        model.save_pretrained(native_dir)

        quantize_config = AutoRoundQuantizeConfig(
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GPTQ,
            desc_act=False,
            iters=1,
            lr=0.05,
            minmax_lr=0.05,
            device="cpu",
            offload_to_disk=False,
        )
        model = GPTQModel.load(native_dir, quantize_config=quantize_config)
        model.quantize(calibration, batch_size=1)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save(tmpdirname)

            with open(f"{tmpdirname}/{QUANT_CONFIG_FILENAME}", "r", encoding="utf-8") as handle:
                file_dict = json.load(handle)

            assert model.quantize_config.to_dict() == file_dict
            assert file_dict["quant_method"] == QUANT_METHOD.AUTO_ROUND
            assert file_dict["checkpoint_format"] == FORMAT.GPTQ

            reloaded = GPTQModel.load(
                tmpdirname,
                device="cpu",
                backend=BACKEND.TORCH,
            )

            versionable = reloaded.quantize_config.meta_get_versionable(META_FIELD_QUANTIZER)
            assert META_QUANTIZER_GPTQMODEL in [producer for producer, _ in versionable]
            for producer, producer_version in versionable:
                if producer == META_QUANTIZER_GPTQMODEL:
                    assert producer_version == __version__

            output = reloaded.model(
                input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
            )
            assert tuple(output.logits.shape) == (1, 4, cfg.vocab_size)
            assert torch.isfinite(output.logits).all()

            compat_quantize_config = {
                "bits": 4,
                "group_size": 16,
                "sym": True,
                "desc_act": False,
            }
            compat_loaded = GPTQModel.load(
                tmpdirname,
                device="cpu",
                quantize_config=compat_quantize_config,
            )
            assert isinstance(compat_loaded.quantize_config, QuantizeConfig)

            del compat_loaded
            del reloaded
            torch_empty_cache()
