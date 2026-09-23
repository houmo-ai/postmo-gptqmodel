"""Loading packed weights must not disable later model initialization."""

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.backend import BACKEND


def _initialization_state():
    return (
        tuple(getattr(torch.nn.init, name) for name in (
            "kaiming_uniform_", "uniform_", "normal_"
        )),
        hasattr(transformers.modeling_utils, "_init_weights"),
        getattr(transformers.modeling_utils, "_init_weights", None),
    )


def test_load_and_resave_restore_torch_initializers(tmp_path):
    original = _initialization_state()
    source, packed, resaved = (tmp_path / name for name in ("source", "packed", "resaved"))
    config = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    Qwen2ForCausalLM(config).save_pretrained(source)
    Qwen2Tokenizer(
        vocab={"<|endoftext|>": 0, "t": 1, "Ġ": 2, **{str(i): i + 3 for i in range(10)}},
        merges=[],
    ).save_pretrained(source)
    quantize_config = QuantizeConfig(
        bits=4, group_size=64, sym=True, desc_act=False,
        act_group_aware=False, device="cpu", offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), quantize_config)
    assert _initialization_state() == original

    calibration = [{"input_ids": torch.randint(1, 128, (1, 32)),
                    "attention_mask": torch.ones(1, 32, dtype=torch.long)} for _ in range(2)]
    model.quantize(calibration, batch_size=1)
    model.save(str(packed))
    assert _initialization_state() == original

    reloaded = GPTQModel.load(str(packed), device="cpu", backend=BACKEND.TORCH)
    assert _initialization_state() == original
    reloaded.save(str(resaved))
    assert _initialization_state() == original
    assert torch.isfinite(torch.nn.Linear(8, 8)(torch.ones(1, 8))).all()
