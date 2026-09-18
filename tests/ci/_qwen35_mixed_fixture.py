"""Reduced-dimension native Qwen MoE with both attention kinds (not a weight slice)."""
import inspect

import pytest
import torch


@pytest.fixture
def native_qwen35_cpu(monkeypatch):
    """Run real Torch kernels regardless of installed optional GPU packages."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

    # Recent Transformers calls module-level decorated functions instead of
    # the instance callables configured by tiny_model for earlier versions.
    # Unwrap the dispatch decorators, retaining the actual Torch algorithms.
    with monkeypatch.context() as patch:
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                     "causal_conv1d_fn", "causal_conv1d_update"):
            kernel = getattr(modeling, name, None)
            if callable(kernel):
                patch.setattr(modeling, name, inspect.unwrap(kernel))
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            yield
        finally:
            torch.set_num_threads(previous_threads)


def tiny_model():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule, Qwen3_5MoeRMSNormGated,
    )

    torch.manual_seed(13)
    config = Qwen3_5MoeConfig(text_config={
        "vocab_size": 256, "hidden_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "linear_key_head_dim": 64, "linear_value_head_dim": 64,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 128,
        "shared_expert_intermediate_size": 128, "tie_word_embeddings": True,
        "pad_token_id": 0, "eos_token_id": 2,
    }, tie_word_embeddings=True, image_token_id=250, video_token_id=251,
       vision_start_token_id=252, vision_end_token_id=253, vision_config={
        "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "out_hidden_size": 128, "num_position_embeddings": 16})
    model = Qwen3_5MoeForConditionalGeneration(config)
    # Use the native CPU reference implementation; FLA is a CUDA-only kernel.
    linear = model.model.language_model.layers[0].linear_attn
    native_norm = Qwen3_5MoeRMSNormGated(linear.head_v_dim, eps=linear.layer_norm_epsilon)
    native_norm.load_state_dict(linear.norm.state_dict())
    linear.norm = native_norm
    linear.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    linear.causal_conv1d_fn = None
    return model


def samples():
    generator = torch.Generator().manual_seed(41)
    return [{"input_ids": torch.randint(3, 250, (1, 32), generator=generator),
             "attention_mask": torch.ones(1, 32, dtype=torch.long)} for _ in range(2)]


def save_source(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tiny_model().save_pretrained(path)
    special = {"<|image_pad|>": 250, "<|video_pad|>": 251, "<|vision_start|>": 252,
               "<|vision_end|>": 253, "<|im_start|>": 254, "<|im_end|>": 255}
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, **{f"t{i}": i for i in range(3, 250)}, **special}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                  eos_token="[EOS]", additional_special_tokens=list(special))
    fast.save_pretrained(path)
    from transformers import Qwen3VLProcessor, Qwen2VLImageProcessorFast, Qwen3VLVideoProcessor
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(patch_size=16), tokenizer=fast,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}{% else %}{% for item in message['content'] %}"
        "{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}"
        "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% endif %}"
        "{% endfor %}{% endif %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    processor.save_pretrained(path)
    return path


def save_image_dataset(path):
    import json
    from PIL import Image

    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, answer in enumerate(("t7", "t9")):
        pixels = ((torch.arange(64 * 64 * 3).reshape(64, 64, 3) * (i + 1) + i * 77) % 256).byte()
        Image.fromarray(pixels.numpy()).save(path / f"image_{i}.png")
        rows.append({"id": str(i), "image": f"image_{i}.png", "text": "t3 t4 t5",
                     "answer": answer, "options": ["t7", "t9"]})
    manifest = path / "images.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return manifest
