"""One-call MoE AutoRound backbone and W5 endpoint calibration on CUDA."""
import json

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from test_minicpm_v47_head import _make_qdq


def test_one_call_moe_autoround_and_w5_head(tmp_path):
    from gptqmodel import GPTQModel
    from gptqmodel.recipes.minicpm_v_4_7 import quantize_minicpm_v_4_7_autoround
    from gptqmodel.quantization.lm_head import HeadQuantizationConfig, inspect_checkpoint_head

    torch.set_num_threads(2)
    source, data = _make_qdq(tmp_path, True)
    text_config = json.loads((source / 'config.json').read_text())
    # The recipe extracts the host's language_model + lm_head. Keep a real
    # VLM host here, with the same hybrid MoE text path and an unused tiny ViT.
    host = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(
        text_config=text_config, vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=4, out_hidden_size=128, patch_size=14, spatial_merge_size=2,
        temporal_patch_size=2, deepstack_visual_indexes=[])))
    state = torch.load(source / 'pytorch_model.bin', weights_only=True)
    host.model.language_model.load_state_dict({k.removeprefix('model.'): v for k,v in state.items()
                                               if k.startswith('model.')})
    host.lm_head.weight.data.copy_(state['lm_head.weight'])
    native = tmp_path / 'native'
    host.save_pretrained(native)
    AutoTokenizer.from_pretrained(source).save_pretrained(native)
    target = tmp_path / 'moe_w5'
    result = quantize_minicpm_v_4_7_autoround(
        model_dir=str(native), output_dir=str(target), dataset=str(data),
        nsamples=2, seqlen=16, iters=2, batch_size=1, low_gpu_mem_usage=False,
        device='cuda:0', device_map='0', head={'bits': 5, 'iters': 2,
        'row_chunk_size': 128, 'batch_tokens': 16})
    assert result.output_dir == str(target)
    assert result.provenance['head']['algorithm'] == 'autoround'
    assert inspect_checkpoint_head(target, HeadQuantizationConfig(bits=5))['actual']['bits'] == 5
    config = json.loads((target / 'config.json').read_text())
    assert config['quantization_config']['meta']['lm_head_quantization']['algorithm'] == 'autoround'
    model = GPTQModel.load(str(target), device='cpu')
    with torch.inference_mode():
        assert torch.isfinite(model.model(torch.ones(1, 16, dtype=torch.long)).logits).all()
    assert not list(tmp_path.glob('moe_w5.backbone-*'))
