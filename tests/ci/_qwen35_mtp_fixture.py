"""Small native Qwen MoE with real MTP FC, gated attention, router and experts."""
import json

import torch
from safetensors.torch import save_file

from _qwen35_mixed_fixture import save_source


def save_mtp_source(path):
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, Qwen35MTP

    save_source(path)
    reader = CheckpointTensors(path)
    config = reader.config
    config["text_config"]["mtp_num_hidden_layers"] = 1
    model = Qwen35MTP(config).half()
    generator = torch.Generator().manual_seed(29)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm" in name:
                value.zero_()
            else:
                value.copy_(torch.randn(value.shape, generator=generator) * 0.03)
    state = {"mtp." + k: v.contiguous() for k, v in model.state_dict().items()}
    filename = "model-mtp-float.safetensors"
    save_file(state, path / filename)
    (path / "model.safetensors").rename(path / "model-main.safetensors")
    mapping = {k: "model-main.safetensors" for k in reader.mapping}
    mapping.update(dict.fromkeys(state, filename))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping, "metadata": {}}))
    (path / "config.json").write_text(json.dumps(config))
    return path
