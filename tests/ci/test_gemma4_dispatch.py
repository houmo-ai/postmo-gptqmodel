"""Exercise real Accelerate dispatch without requiring a GPU."""
import pytest
import torch
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

from gptqmodel.utils.model import simple_dispatch_model


@pytest.mark.parametrize('cpu_offload', [False, True])
def test_dispatch_preserves_shared_kv_identity_and_generation(cpu_offload):
    torch.manual_seed(13)
    config = Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=64,
        layer_types=['sliding_attention', 'full_attention'] * 2,
        num_kv_shared_layers=2, hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128, sliding_window=16, max_position_embeddings=64,
        attention_k_eq_v=False,
    )
    config._attn_implementation = 'eager'
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=config)).eval()
    inputs = {'input_ids': torch.tensor([[2, 3, 4, 5]])}
    with torch.no_grad():
        expected_logits = model(**inputs, use_cache=False).logits
        expected_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    # Explicit indexed CPU devices use the same AlignDevicesHook branch as
    # a module-level CUDA map, rather than the single-root/all-CPU shortcut.
    mapping = {f'model.language_model.layers.{i}': 'cpu' if cpu_offload and i % 2 == 0 else 'cpu:0'
               for i in range(4)}
    simple_dispatch_model(model, mapping)
    with torch.no_grad():
        actual_logits = model(**inputs, use_cache=False).logits
        actual_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
