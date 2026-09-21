"""Synthetic small native GLM KDA+DSA+mHC+MoE+vision, not a release-weight block."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.glm5_next.configuration_glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
)


def tiny_glm53_config():
    text = Glm5NextTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_head_dim=64,
        linear_num_heads=2,
        linear_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_group=1,
        topk_group=1,
        index_n_heads=8,
        index_head_dim=16,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        mhc=True,
        mla_use_nope=True,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=2,
        swiglu_limit=1.0,
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=128,
        patch_size=2,
        image_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        projection_intermediate_size=128,
    )
    return Glm5NextConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=240,
        video_token_id=241,
        image_start_token_id=242,
        image_end_token_id=243,
        video_start_token_id=244,
        video_end_token_id=245,
        tie_word_embeddings=False,
    )


def write_tiny_glm53(path):
    torch.manual_seed(489)
    torch.set_num_threads(2)
    model = Glm5NextForConditionalGeneration(tiny_glm53_config()).eval()
    model.config.dtype = torch.bfloat16
    model.save_pretrained(path)
    tok = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[EOS]": 2,
                **{f"word{i}": i + 3 for i in range(200)},
            },
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(path)
    return model


def calibration_texts(count=16):
    return [
        " ".join(f"word{(i * 17 + j) % 200}" for j in range(32)) for i in range(count)
    ]
