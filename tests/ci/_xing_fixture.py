"""Synthetic small Xing models using bundled native remote code.

XING4_0_MODEL_DIR may override the source; no 29B weight shards are read.
These randomly initialized models are not original-weight block fixtures.
"""

import os
from pathlib import Path

import torch


def tiny_xing_model(**overrides):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from gptqmodel.models.definitions.xing4_0_mla import patch_loaded_xing_classes

    source = Path(
        os.environ.get("XING4_0_MODEL_DIR")
        or Path(__file__).resolve().parent / "fixtures/xing_native"
    ).resolve()
    for name in ("config.json", "configuration_xing4_0.py", "modeling_xing4_0.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    config = AutoConfig.from_pretrained(str(source), trust_remote_code=True)
    values = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        qk_head_dim=64,
        v_head_dim=32,
        head_dim=32,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        use_cache=False,
        attention_dropout=0.0,
        hc_sinkhorn_iters=3,
        rope_scaling=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    values.update(overrides)
    for key, value in values.items():
        setattr(config, key, value)
    get_class_from_dynamic_module("modeling_xing4_0.Xing4_0ForCausalLM", str(source))
    patch_loaded_xing_classes()
    torch.manual_seed(31)
    return AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager", dtype=torch.float32
    ).eval()


def save_tiny_xing(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    model = tiny_xing_model().to(torch.float16)
    from gptqmodel.models.definitions.xing4_0_mla import refresh_xing_rotary_buffers

    refresh_xing_rotary_buffers(model)
    # Values that would change if a global BF16/FP16 cast rounded this router
    # parameter despite the native model's strict FP32 declaration.
    model.model.layers[1].mlp.gate.e_score_correction_bias.data = torch.tensor(
        [0.1234567, -0.2345678], dtype=torch.float32
    )
    model.save_pretrained(path)
    tokenizer = Tokenizer(
        WordLevel(vocab={str(i): i for i in range(128)}, unk_token="3")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    ).save_pretrained(path)
    return model
