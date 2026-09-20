"""Real hybrid Qwen3.5 quantization/packed reload using a small random model.

This tests the solver and serialization, not pretrained 27B task accuracy.
"""
import json

import pytest
import torch
from safetensors.torch import load_file


@pytest.mark.cuda
@pytest.mark.parametrize('named_preset', [True, False], ids=['named-preset', 'explicit-backend-mode'])
def test_hybrid_qwen_xh_hessian_quantize_and_reload(tmp_path, named_preset):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers import Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    assert torch.cuda.is_available(), 'xh_hessian quant lane requires CUDA'
    torch.manual_seed(42)
    text_config = dict(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=64, linear_value_head_dim=64,
        layer_types=['linear_attention', 'full_attention'], max_position_embeddings=256,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    source, output = tmp_path / 'source', tmp_path / 'quantized'
    config = Qwen3_5Config(text_config=text_config, vision_config=dict(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=4,
        out_hidden_size=128, patch_size=14, num_position_embeddings=64,
    ))
    model = Qwen3_5ForConditionalGeneration(config).eval().to(torch.bfloat16)
    model.save_pretrained(source)
    tokenizer = Tokenizer(WordLevel(vocab={str(i): i for i in range(128)}, unk_token='3'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='0', bos_token='1',
                                       eos_token='2', unk_token='3')
    Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(), tokenizer=tokenizer,
                     video_processor=Qwen3VLVideoProcessor()).save_pretrained(source)
    calibration = tmp_path / 'calibration.jsonl'
    calibration.write_text('\n'.join(json.dumps({'text': ' '.join(str((i+j)%120+4) for i in range(96))})
                                    for j in range(32)) + '\n')
    options = ({'data_type': 'int_sym_xh_hessian'} if named_preset else {
        'data_type': 'int_sym_xh', 'optimization_mode': 'xh_hessian',
        'xh_hessian_config': {'candidates': 5, 'validation_samples': 2, 'validation_interval': 2},
    })
    nsamples, validation_samples = (24, 16) if named_preset else (8, 2)
    quantize_qwen35_autoround(
        model_dir=str(source), output_dir=str(output), topology='dense',
        **options, nsamples=nsamples, batch_size=2, seqlen=32, iters=4,
        dataset=str(calibration), device='cuda:0', low_gpu_mem_usage=True,
        env={'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'},
    )
    report = json.loads((output / 'xh_hessian_report.json').read_text())['records']
    matrices = [r for r in report if 'module' in r]
    blocks = [r for r in report if 'block' in r]
    assert len(blocks) == 2 and len(matrices) >= 10
    assert all(r['training_samples'] == nsamples - validation_samples
               and r['validation_samples'] == validation_samples for r in blocks)
    assert all(r['validation_mse'] <= r['initial_validation_mse'] for r in blocks)
    assert all(r['tokens'] == (nsamples - validation_samples) * 32 for r in matrices)
    assert all(r['searched_score'] <= r['xh_initial_score'] for r in matrices)
    packed = load_file(output / 'model.safetensors')
    saved_config = json.loads((output / 'config.json').read_text())['quantization_config']
    assert saved_config['optimization_mode'] == 'xh_hessian'
    assert saved_config['xh_hessian_config']['validation_samples'] == validation_samples
    assert saved_config['data_type'] == 'int_sym_xh'
    if named_preset:
        assert saved_config['requested_data_type'] == 'int_sym_xh_hessian'
        assert saved_config['xh_hessian_config']['candidates'] == 33
    for name, value in packed.items():
        if name.endswith('.scales'):
            assert torch.isfinite(value).all() and (value > 0).all()
        if name.endswith('.qzeros'):
            assert (value == 0x77777777).all()
    assert any('.0.linear_attn.' in n and n.endswith('.qweight') for n in packed)
    assert any('.1.self_attn.' in n and n.endswith('.qweight') for n in packed)

    loaded = GPTQModel.load(str(output), device='cuda:0', backend=BACKEND.TORCH, dtype=torch.float16)
    inputs = torch.tensor([[4, 5, 6, 7]], device='cuda:0')
    with torch.no_grad():
        logits = loaded.model(input_ids=inputs, use_cache=False).logits
    assert torch.isfinite(logits).all() and logits.shape == (1, 4, 128)
    # Independently decode saved GPTQ v1 bytes into plain Linear layers. This
    # compares the actual loader/kernel with a format reference, including the
    # stored-zero-plus-one convention and G64 ordering.
    checked = 0
    for name, module in list(loaded.model.named_modules()):
        if not hasattr(module, 'qweight'):
            continue
        qweight = packed[name + '.qweight'].to(torch.int64)
        qzeros = packed[name + '.qzeros'].to(torch.int64)
        scales = packed[name + '.scales']
        shifts = torch.arange(8) * 4
        q = ((qweight[:, None, :] >> shifts[None, :, None]) & 15).reshape(-1, qweight.shape[1])
        z = (((qzeros[:, :, None] >> shifts[None, None, :]) & 15) + 1).reshape(scales.shape)
        idx = packed[name + '.g_idx'].long()
        weight = ((q - z[idx]).float() * scales[idx]).T.contiguous().half().to('cuda:0')
        assert torch.equal(module.qweight.cpu(), packed[name + '.qweight'])
        assert torch.equal(module.scales.cpu(), scales)
        reference = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=module.bias is not None,
                                    device='cuda:0', dtype=torch.float16)
        with torch.no_grad():
            reference.weight.copy_(weight)
            if module.bias is not None:
                reference.bias.copy_(module.bias)
            probe = torch.randn(3, weight.shape[1], device='cuda:0', dtype=torch.float16)
            torch.testing.assert_close(module(probe), reference(probe), rtol=0, atol=0)
        loaded.model.set_submodule(name, reference)
        checked += 1
    assert checked == len(matrices)
    with torch.no_grad():
        reference_logits = loaded.model(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(reference_logits, logits, rtol=0, atol=0)
