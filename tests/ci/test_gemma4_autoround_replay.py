"""Real Gemma decoder replay: mixed RoPE, PLE, donor/consumer and sample order."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel


@pytest.fixture(autouse=True)
def restore_quantization_initializers():
    initializers = {name: getattr(torch.nn.init, name) for name in
                    ('normal_', 'uniform_', 'kaiming_uniform_')}
    missing = object()
    init_weights = getattr(transformers.modeling_utils, '_init_weights', missing)
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


def _script():
    path = Path(__file__).resolve().parents[2] / 'third_party/auto-round/scripts_gemma4/quantize.py'
    spec = importlib.util.spec_from_file_location('gemma4_quant_replay_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model():
    torch.manual_seed(17)
    config = Gemma4TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=64,
        layer_types=['sliding_attention', 'full_attention'] * 2,
        num_kv_shared_layers=2, hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=128, sliding_window=16, max_position_embeddings=64,
        attention_k_eq_v=False,
    )
    config._attn_implementation = 'eager'
    return Gemma4TextModel(config).eval()


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 0, 1]])
@torch.no_grad()
def test_replay_matches_real_layers_and_keeps_fp_quant_banks_separate(order):
    from auto_round.compressors import LLMCompressor

    model = _model()
    wrapper = SimpleNamespace(model=SimpleNamespace(language_model=model),
                              config=SimpleNamespace(text_config=model.config))
    replay = _script().patch_position_embeddings(wrapper, seqlen=12, device='cpu')
    # Exercise the actual classmethod sampler and output batching without running
    # optimization here. The quant lane covers optimization/save/reload.
    compressor = object.__new__(LLMCompressor)
    compressor.nblocks = 1
    compressor.batch_size = 2
    compressor.batch_dim = 0
    compressor.seqlen = 6
    compressor.shared_cache_keys = ()
    compressor.xh_hessian_config = None
    compressor.amp = False
    compressor.amp_dtype = torch.float32
    compressor.low_gpu_mem_usage = False
    from auto_round.compressors.utils import block_forward
    compressor.block_forward = block_forward
    replay.attach(compressor)
    initial, references, masks = [], [], []

    def capture(module, args, kwargs, output):
        if module is model.layers[0]:
            initial.extend(args[0].detach().split(1))
            mask = kwargs['attention_mask']
            masks.extend(mask.detach().split(1))
        references.append(output.detach())

    handles = [layer.register_forward_hook(capture, with_kwargs=True) for layer in model.layers]
    tokens = torch.tensor([[2, 4, 6, 8, 10, 12], [2, 3, 5, 7, 9, 11], [2, 13, 14, 15, 16, 17]])
    model(tokens, use_cache=False)
    for handle in handles:
        handle.remove()
    assert len(replay.per_layer_inputs) == 3
    others = {'positional_inputs': (), 'attention_mask': masks,
              'per_layer_input': None, 'position_embeddings': None, 'past_key_values': None}
    inputs = initial
    for index, layer in enumerate(model.layers):
        replay._first_outputs = True
        outputs = compressor._get_block_outputs(layer, inputs, others, 2, torch.device('cpu'), 'cpu')
        torch.testing.assert_close(torch.cat(outputs), references[index], atol=1e-6, rtol=1e-5)
        inputs = outputs
    float_bank = {kind: {i: tuple(t.clone() for t in pair) for i, pair in bank.items()}
                  for kind, bank in replay.kv['float'].items()}
    # A distinct propagated donor stream must not replace the FP teacher cache.
    with torch.no_grad():
        model.layers[0].self_attn.k_proj.weight.add_(0.1)
    replay.phase = 'quant'
    inputs = initial
    for layer in model.layers:
        replay._first_outputs = False
        inputs = compressor._get_block_outputs(layer, inputs, others, 2, torch.device('cpu'), 'cpu')
    for kind, bank in float_bank.items():
        for sample, pair in bank.items():
            for part in (0, 1):
                torch.testing.assert_close(replay.kv['float'][kind][sample][part], pair[part], rtol=0, atol=0)
    assert not torch.equal(replay.kv['float']['sliding_attention'][0][0],
                           replay.kv['quant']['sliding_attention'][0][0])
    # Shuffle the real sampler and check both KV and the layer-specific PLE.
    sampled, _ = compressor._sampling_inputs(initial, others, order, 6)
    ple, shared = replay.inputs_for_layer(2, sampled, None, None)
    torch.testing.assert_close(ple, torch.cat([replay.per_layer_inputs[i][:, :, 2] for i in order]))
    torch.testing.assert_close(shared['sliding_attention'][0],
                              torch.cat([replay.kv['quant']['sliding_attention'][i][0] for i in order]))
    replay.kv['quant']['sliding_attention'].pop(order[0])
    with pytest.raises(RuntimeError, match='KV donor missing'):
        replay.inputs_for_layer(2, sampled, None, None)
    replay.close()
    assert '_sampling_inputs' not in vars(compressor)
    assert 'forward' not in vars(model.layers[0])
    assert 'project_per_layer_inputs' not in vars(model)


@pytest.mark.parametrize('enable_quanted_input', [True, False])
def test_real_autoround_optimize_pack_and_reload(tmp_path, enable_quanted_input):
    from auto_round.compressors import LLMCompressor
    from gptqmodel import GPTQModel
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, PreTrainedTokenizerFast

    config = Gemma4Config(text_config=_model().config, vision_config=None, audio_config=None)
    model = Gemma4ForConditionalGeneration(config).eval()
    backend = Tokenizer(WordLevel({'[UNK]': 0, **{f't{i}': i for i in range(1, 128)}}, unk_token='[UNK]'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[UNK]')
    dataset = [{'input_ids': torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])},
               {'input_ids': torch.tensor([[2, 9, 8, 7, 6, 5, 4, 3]])}]
    # Install before constructing AutoRound, as the real CLI does.
    replay = _script().patch_position_embeddings(model, 8, 'cpu')
    compressor = LLMCompressor(
        model=model, tokenizer=tokenizer, dataset=dataset, device_map='cpu', bits=4,
        group_size=64, data_type='int_sym_gptq', iters=2, nsamples=2, seqlen=8,
        batch_size=1, amp=False, low_cpu_mem_usage=False,
        enable_quanted_input=enable_quanted_input,
        layer_config={'per_layer_input_gate': {'bits': 16}, 'per_layer_projection': {'bits': 16}},
    )
    replay.attach(compressor)
    try:
        compressor.quantize()
        assert set(replay.kv['float']) == {'sliding_attention', 'full_attention'}
        for bank in replay.kv['float'].values():
            assert set(bank) == {0, 1}
        if enable_quanted_input:
            assert set(replay.kv['quant']) == {'sliding_attention', 'full_attention'}
    finally:
        replay.close()
    compressor.save_quantized(str(tmp_path), format='auto_gptq')
    saved = {}
    for shard in tmp_path.glob('*.safetensors'):
        saved.update(load_file(shard))
    quantized = {key.removesuffix('.qweight') for key in saved if key.endswith('.qweight')}
    assert len(quantized) == 24  # 7 + 7 donor projections, 5 + 5 consumer projections
    for layer in range(4):
        prefix = f'model.language_model.layers.{layer}'
        assert prefix + '.self_attn.q_proj' in quantized
        assert (prefix + '.self_attn.k_proj' in quantized) == (layer < 2)
        assert (prefix + '.self_attn.v_proj' in quantized) == (layer < 2)
        assert prefix + '.per_layer_input_gate.weight' in saved
        assert prefix + '.per_layer_projection.weight' in saved
    reloaded = GPTQModel.load(str(tmp_path), device='cpu', dtype=torch.float32)
    state = reloaded.model.state_dict()
    for name in quantized:
        assert saved[name + '.qweight'].dtype == torch.int32
        scales = saved[name + '.scales']
        assert torch.isfinite(scales).all() and (scales > 0).all()
        # Integer payload and scales survive the real loader unchanged. GPTQ-v1
        # qzeros may be normalized to v2 by the loader, so do not compare them raw.
        torch.testing.assert_close(state[name + '.qweight'], saved[name + '.qweight'], rtol=0, atol=0)
        torch.testing.assert_close(state[name + '.scales'].float(), scales.float(), rtol=0, atol=0)
    with torch.no_grad():
        logits = reloaded.model(**dataset[0], use_cache=False).logits
    assert logits.shape == (1, 8, 128)
    assert torch.isfinite(logits).all()
