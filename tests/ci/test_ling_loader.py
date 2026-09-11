"""A cached original Ling class must reconstruct a later absorbed checkpoint."""
import importlib.util
import json
import shutil
from pathlib import Path

import torch


def test_same_basename_checkpoints_keep_config_selected_mla_layout(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, modeling_utils

    from gptqmodel.models.definitions.ling_3_flash import load_ling_3_flash_config
    from gptqmodel.models.definitions.ling3_flash_mla import XH2_MLA_LAYOUT, finalize_ling_mla_checkpoint

    # Keep test order independent of GPTQModel writer's global init patches.
    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(modeling_utils, "_init_weights", getattr(modeling_utils, "_init_weights", True), raising=False)
    helper_path = Path(__file__).resolve().parent / "test_ling3_flash_autoround.py"
    spec = importlib.util.spec_from_file_location("ling_source_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original, absorbed = tmp_path / "original/out", tmp_path / "absorbed/out"
    helper._save_real_tiny_ling(original, 0, num_attention_heads=32, num_key_value_heads=32,
                               qk_rope_head_dim=64, qk_head_dim=96)
    absorbed.mkdir(parents=True)
    for name in ("config.json", "configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        shutil.copy2(original / name, absorbed / name)
    config = json.loads((absorbed / "config.json").read_text())
    config["ling_mla_layout"] = XH2_MLA_LAYOUT
    (absorbed / "config.json").write_text(json.dumps(config))
    finalize_ling_mla_checkpoint(absorbed)

    # Seed AutoModel's registry with the unabsorbed class before it sees the
    # checkpoint whose model code has the absorption hook appended.
    previous_class = None
    for path, should_absorb in ((original, False), (absorbed, True), (original, False)):
        config = load_ling_3_flash_config(str(path))
        if previous_class is not None:
            assert type(config) is previous_class, "fixture must exercise the shared HF config-class cache"
        previous_class = type(config)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
        attention = model.model.layers[1].attention
        assert hasattr(attention, "q_absorbed_proj") == should_absorb
        assert hasattr(attention, "kv_b_proj") != should_absorb
        assert all(parameter.is_meta for parameter in model.parameters())
