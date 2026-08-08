from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.nn as nn

from auto_round.export.export_to_autoround.export import save_quantized_as_autoround
from auto_round.schemes import QuantizationScheme


def test_gptqmodel_export_tolerates_layer_config_without_in_blocks():
    model = nn.Module()
    model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
    model.config = SimpleNamespace()
    scheme = QuantizationScheme(
        bits=4,
        group_size=64,
        sym=True,
        data_type="int_sym_gptq",
        act_bits=16,
    )
    serialization_dict = {
        **asdict(scheme),
        "to_quant_block_names": None,
        "regex_config": None,
    }
    layer_config = {
        "outside_layer": {
            **asdict(scheme),
            "bits": 8,
        }
    }

    save_quantized_as_autoround(
        output_dir=None,
        model=model,
        layer_config=layer_config,
        backend="auto_round:gptqmodel",
        serialization_dict=serialization_dict,
    )

    assert model.config.quantization_config["extra_config"]["outside_layer"]["bits"] == 8
