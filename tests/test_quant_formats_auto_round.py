# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, __version__
from gptqmodel.quantization import FORMAT, QUANT_CONFIG_FILENAME
from gptqmodel.quantization.config import (
    META_FIELD_QUANTIZER,
    META_QUANTIZER_GPTQMODEL,
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
    tokenizer_backend = Tokenizer(WordLevel(vocab={str(index): index for index in range(128)}, unk_token="3"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    )
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
    return cfg, model, tokenizer, calibration


def test_auto_round_quantize_config_roundtrips_on_saved_model():
    torch.manual_seed(0)
    cfg, model, tokenizer, calibration = _build_tiny_llama()

    with tempfile.TemporaryDirectory() as native_dir:
        model.save_pretrained(native_dir)
        tokenizer.save_pretrained(native_dir)

        quantize_config = QuantizeConfig(
            auto_round_enabled=True,
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GPTQ,
            data_type="int_sym_xh",
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
            assert file_dict["quant_method"] == QUANT_METHOD.GPTQ
            assert file_dict["auto_round_enabled"] is True
            assert file_dict["checkpoint_format"] == FORMAT.GPTQ
            assert file_dict["data_type"] == "int_sym_xh"

            with open(f"{tmpdirname}/config.json", "r", encoding="utf-8") as handle:
                model_config_dict = json.load(handle)
            assert model_config_dict["quantization_config"]["data_type"] == "int_sym_xh"

            reloaded = GPTQModel.load(
                tmpdirname,
                device="cpu",
                backend=BACKEND.TORCH,
            )

            versionable = reloaded.quantize_config.meta_get_versionable(META_FIELD_QUANTIZER)
            assert reloaded.quantize_config.data_type == "int_sym_xh"
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

            with tempfile.TemporaryDirectory() as resaved_dir:
                reloaded.save(resaved_dir)
                with open(f"{resaved_dir}/{QUANT_CONFIG_FILENAME}", "r", encoding="utf-8") as handle:
                    resaved_quant_config = json.load(handle)
                with open(f"{resaved_dir}/config.json", "r", encoding="utf-8") as handle:
                    resaved_model_config = json.load(handle)
                assert resaved_quant_config["data_type"] == "int_sym_xh"
                assert resaved_model_config["quantization_config"]["data_type"] == "int_sym_xh"

                reloaded_again = GPTQModel.load(
                    resaved_dir,
                    device="cpu",
                    backend=BACKEND.TORCH,
                )
                assert reloaded_again.quantize_config.data_type == "int_sym_xh"
                output_again = reloaded_again.model(
                    input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                    attention_mask=torch.ones((1, 4), dtype=torch.long),
                )
                assert tuple(output_again.logits.shape) == (1, 4, cfg.vocab_size)
                assert torch.isfinite(output_again.logits).all()
                del reloaded_again

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
