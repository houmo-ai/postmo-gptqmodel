"""Dense MTP through the public full-model recipe, including packed shared head."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize("algorithm", ["autoround", "gptq"])
def test_dense_full_recipe_and_reuse(tmp_path, algorithm):
    from _qwen35_mixed_fixture import save_source
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, Qwen35MTP
    from gptqmodel.recipes.qwen35_mtp import quantize_qwen35_mtp

    torch.manual_seed(71)
    source = save_source(tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    for field in ("num_experts", "num_experts_per_tok", "moe_intermediate_size",
                  "shared_expert_intermediate_size", "decoder_sparse_step", "mlp_only_layers"):
        config["text_config"].pop(field, None)
    config["text_config"].update(model_type="qwen3_5_text", intermediate_size=256, mtp_num_hidden_layers=1)
    config["model_type"] = "qwen3_5"
    config.pop("architectures", None)
    native = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(config)).half()
    native.save_pretrained(source)
    reader = CheckpointTensors(source)
    model = Qwen35MTP(reader.config).half()
    with torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(torch.zeros_like(value) if "norm" in name else torch.randn_like(value) * .03)
    state = {"mtp." + name: value.contiguous() for name, value in model.state_dict().items()}
    save_file(state, source / "mtp.safetensors")
    mapping = dict(reader.mapping)
    mapping.update(dict.fromkeys(state, "mtp.safetensors"))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping, "metadata": {}}))
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": " ".join(f"t{i}" for i in range(3, 100))}) for _ in range(3)) + "\n")
    options = dict(model_dir=str(source), output_dir=str(tmp_path / "out"), topology="dense",
                   nsamples=2, seqlen=32, batch_size=1, device="cuda:0", head={"bits": 5, "iters": 2},
                   mtp={"block_bits": 5, "iters": 2, "batch_tokens": 32, "validation_nsamples": 1})
    if algorithm == "autoround":
        from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround
        result = quantize_qwen35_autoround(**options, dataset=str(data), iters=2,
                                          data_type="int_sym_xh", device_map="0")
    else:
        from gptqmodel.recipes.qwen35 import quantize_qwen35
        result = quantize_qwen35(**options, calibration_jsonl=str(data), device_map=None,
                                 check_quant_vision_demo=False, skip_inference=True)
    reader = CheckpointTensors(result.output_dir)
    assert reader.config["mtp_quantization"]["scope"] == "dense_block"
    assert reader.config["mtp_quantization"]["block_bits"] == 5
    assert reader.config["mtp_quantization"]["algorithm"] == algorithm
    assert "lm_head.qweight" in reader.mapping
    assert "mtp.fc.weight" in reader.mapping
    assert len([k for k in reader.mapping if k.startswith("mtp.") and k.endswith(".qweight")]) == 7
    validation = result.provenance["mtp_stage"]["validation"]
    assert validation["logical_bits"] == 5 and "draft_fp_to_w5_kl" in validation
    loaded = Qwen35MTP.from_pretrained(result.output_dir).cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, 128, dtype=torch.float16, device="cuda:0")
        assert torch.isfinite(loaded(x, x)).all()
    reuse = quantize_qwen35_mtp(model_dir=result.output_dir, mtp={"block_bits": 5})
    assert reuse.method == "skip" and Path(reuse.output_dir) == Path(result.output_dir)
