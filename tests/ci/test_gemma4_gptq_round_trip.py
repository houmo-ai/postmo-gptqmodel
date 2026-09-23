"""Real GPTQ calibration, pack/reload and shared-KV device dispatch."""
import json

import pytest
import torch
import transformers
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import simple_dispatch_model


@pytest.fixture
def restore_quantization_globals():
    initializers = {name: getattr(torch.nn.init, name) for name in
                    ('normal_', 'uniform_', 'kaiming_uniform_')}
    missing = object()
    init_weights = getattr(transformers.modeling_utils, '_init_weights', missing)
    num_threads = torch.get_num_threads()
    try:
        yield
    finally:
        for name, initializer in initializers.items():
            setattr(torch.nn.init, name, initializer)
        if init_weights is missing:
            if hasattr(transformers.modeling_utils, '_init_weights'):
                del transformers.modeling_utils._init_weights
        else:
            transformers.modeling_utils._init_weights = init_weights
        torch.set_num_threads(num_threads)


def test_gemma4_gptq_shared_kv_round_trip(tmp_path, restore_quantization_globals):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    cfg = Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=64,
        layer_types=['sliding_attention', 'full_attention'] * 2, num_kv_shared_layers=2,
        hidden_size_per_layer_input=16, vocab_size_per_layer_input=128, sliding_window=16,
        max_position_embeddings=64, attention_k_eq_v=False,
    )
    cfg._attn_implementation = 'eager'
    native = Gemma4ForConditionalGeneration(Gemma4Config(text_config=cfg, vision_config=None, audio_config=None)).eval()
    fp, quant = tmp_path/'fp', tmp_path/'quant'
    native.save_pretrained(fp)
    backend = Tokenizer(WordLevel({'[UNK]':0, **{f't{i}':i for i in range(1,128)}}, unk_token='[UNK]'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[UNK]')
    tokenizer.save_pretrained(fp)
    calibration = [{'input_ids':torch.randint(1,128,(1,32)), 'attention_mask':torch.ones(1,32,dtype=torch.long)}
                   for _ in range(4)]
    qcfg = QuantizeConfig(bits=4, group_size=64, sym=True, desc_act=False, act_group_aware=False,
                         device='cpu', offload_to_disk=False)
    model = GPTQModel.load(str(fp), qcfg, dtype=torch.float32)
    model.quantize(calibration, batch_size=1)
    assert 'use_cache' not in model.model.config.to_dict()
    model.save(str(quant))
    assert 'use_cache' not in json.loads((quant/'config.json').read_text())
    packed = {}
    for path in quant.glob('*.safetensors'):
        packed.update(load_file(path))
    payloads = {k:v for k,v in packed.items() if k.endswith(('.qweight','.scales'))}
    assert len([k for k in payloads if k.endswith('.qweight')]) == 24
    loaded = GPTQModel.load(str(quant), device='cpu', backend=BACKEND.TORCH, dtype=torch.float32)
    for key,value in payloads.items():
        torch.testing.assert_close(loaded.model.state_dict()[key],value,rtol=0,atol=0,check_dtype=False)
    inputs = calibration[0]
    with torch.no_grad():
        reference = loaded.model(**inputs, use_cache=False).logits
        reference_ids = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
    mapping = {f'model.language_model.layers.{i}':'cpu:0' for i in range(4)}
    simple_dispatch_model(loaded.model,mapping)
    with torch.no_grad():
        actual = loaded.model(**inputs,use_cache=False).logits
        actual_ids = loaded.generate(**inputs,max_new_tokens=3,do_sample=False)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    torch.testing.assert_close(actual_ids,reference_ids,rtol=0,atol=0)
