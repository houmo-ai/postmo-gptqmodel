"""CUDA warmup followed by real AutoRound without a workspace export."""
import os
import subprocess
import sys
import textwrap


def test_cuda_warmup_then_autoround_preserves_workspace_and_finite_outputs():
    # A fresh process is required: cuBLAS caches workspace allocations per
    # handle/stream, and the host test process can already have initialized it.
    program = textwrap.dedent("""
        import os
        from types import SimpleNamespace
        import torch
        from auto_round import AutoRound
        from transformers import LlamaConfig, LlamaForCausalLM

        assert torch.cuda.is_available(), 'quant lane requires CUDA'
        assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        )
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config).eval().to(torch.bfloat16)
        # Other GPTQModel tests can disable nn.init. Give this numerical fixture
        # explicit finite weights independently of those process-global hooks.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.02)

        # Establish PyTorch's default cuBLAS workspace before AutoRound starts.
        x = torch.randn(1536, 2048, device='cuda')
        w = torch.randn(512, 2048, device='cuda')
        warmup = x @ w.T
        torch.cuda.synchronize()
        del x, w, warmup
        tokens = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19] * 2])
        calibration = [{'input_ids': tokens.clone(), 'attention_mask': torch.ones_like(tokens)}
                       for _ in range(2)]
        compressor = AutoRound(
            model, tokenizer=SimpleNamespace(), bits=4, group_size=64,
            data_type='int_sym_xh', dataset=calibration, iters=2, nsamples=2,
            seqlen=16, batch_size=1, device_map='0', low_gpu_mem_usage=True,
            low_cpu_mem_usage=False, enable_torch_compile=False,
        )
        assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
        compressor.quantize()
        scales = [module.scale for module in model.modules()
                  if torch.is_tensor(getattr(module, 'scale', None))]
        assert scales, 'AutoRound must actually quantize decoder projections'
        assert all(bool(torch.isfinite(scale).all()) for scale in scales)
        assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
        with torch.no_grad():
            logits = model.to('cuda')(tokens.to('cuda'), use_cache=False).logits
        assert torch.isfinite(logits).all()
        assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
        print('workspace unchanged; real BF16 AutoRound and logits finite')
    """)
    environment = os.environ.copy()
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run(
        [sys.executable, "-c", program], env=environment,
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
