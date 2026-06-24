import hashlib
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils.missing_tensors import copy_missing_tensors_from_source


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copy_missing_tensors_uses_sidecar_without_rewriting_target_shard(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "mtp.fc.weight": torch.ones(2, 3, dtype=torch.float32),
            "mtp.layers.0.self_attn.q_proj.weight": torch.full((2, 3), 2.0),
        },
        source / "model.safetensors",
        metadata={"format": "pt"},
    )

    target_shard = target / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.language_model.layers.0.self_attn.q_proj.qweight": torch.tensor([[1, 2, 3]], dtype=torch.int32),
            "model.language_model.layers.0.self_attn.q_proj.qzeros": torch.tensor([[4, 5, 6]], dtype=torch.int32),
        },
        target_shard,
        metadata={"format": "pt"},
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.language_model.layers.0.self_attn.q_proj.qweight": target_shard.name,
                    "model.language_model.layers.0.self_attn.q_proj.qzeros": target_shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    before_hash = _sha256(target_shard)

    copied = copy_missing_tensors_from_source(str(source), str(target))

    assert copied == 2
    assert _sha256(target_shard) == before_hash

    sidecar = target / "missing_tensors_passthrough.safetensors"
    assert sidecar.is_file()
    with safe_open(sidecar, framework="pt", device="cpu") as f:
        assert set(f.keys()) == {"mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight"}
        assert torch.equal(f.get_tensor("mtp.fc.weight"), torch.ones(2, 3))

    index = json.loads((target / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"]["model.language_model.layers.0.self_attn.q_proj.qweight"] == target_shard.name
    assert index["weight_map"]["mtp.fc.weight"] == sidecar.name
    assert index["weight_map"]["mtp.layers.0.self_attn.q_proj.weight"] == sidecar.name
