# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[1]
repo_str = str(repo_root)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from gptqmodel.quantization.mix_precision.metrics import MetricFactory
from gptqmodel.quantization.mix_precision.mixsearch_config import validate_mix_precision_config
from gptqmodel.quantization.mix_precision.search_engine import MixPrecisionSearchEngine


class _ToyModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        vocab_size = 3
        logits = F.one_hot(input_ids % vocab_size, num_classes=vocab_size).float()
        return {"logits": logits}


def test_mix_precision_config_accepts_ppl_metric():
    validate_mix_precision_config(
        {
            "enabled": True,
            "weight_bits": [4, 8],
            "policy": "topk",
            "topk": 0.1,
            "metric": "ppl",
        }
    )


def test_ppl_metric_uses_quant_minus_fp_delta():
    metric = MetricFactory.create("ppl")
    fp_ppl = torch.tensor([10.0])
    qt_ppl = torch.tensor([12.5])

    sensitivity = metric.compute(fp_ppl, qt_ppl)

    assert sensitivity == 2.5


def test_search_engine_prepares_batch_ppl_from_logits():
    config = {
        "enabled": True,
        "weight_bits": [4, 8],
        "policy": "topk",
        "topk": 0.1,
        "metric": "ppl",
    }
    engine = MixPrecisionSearchEngine(
        model=_ToyModel(),
        config=config,
        calibration_data=[],
    )

    batch = {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }
    logits = torch.tensor(
        [[[4.0, 1.0, 0.0], [0.5, 3.5, 0.5], [0.1, 0.2, 5.0]]],
        dtype=torch.float32,
    )

    ppl = engine._prepare_output_for_metric(logits, batch, move_to_cpu=True)

    shift_logits = logits[:, :-1, :]
    shift_labels = batch["input_ids"][:, 1:]
    expected_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    expected_ppl = torch.exp(expected_loss).reshape(1)

    assert ppl.device.type == "cpu"
    assert torch.allclose(ppl, expected_ppl.cpu(), atol=1e-6)
