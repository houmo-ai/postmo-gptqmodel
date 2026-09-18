import json
import math

import pytest
import torch

from gptqmodel.quantization.auto_mixed_precision import (
    AutoMixedPrecisionConfig, _records, discover_units, dynamic_overrides,
    output_metrics, parse_auto_mixed_precision, search_precision, select_assignment,
)
from _qwen35_mixed_fixture import native_qwen35_cpu, samples, tiny_model


@pytest.mark.parametrize("value", ["true", 1, {"max_kl": float("nan")}, {"max_kl": -1},
    {"candidates": [4, 8], "initial_bits": 5}, {"candidates": [4, 5]},
    {"candidates": [4, 5, 8, 8]}, {"nsamples": True}, {"positions_per_sample": 512},
    {"max_trials": 0}, {"unknown": 1}, {"dataset": []}])
def test_invalid_config(value):
    with pytest.raises(ValueError):
        parse_auto_mixed_precision(value)


def test_legacy_and_default_config():
    assert parse_auto_mixed_precision(None) is None
    assert parse_auto_mixed_precision(False) is None
    assert parse_auto_mixed_precision(True) == parse_auto_mixed_precision({})
    assert parse_auto_mixed_precision(True).candidates == (4, 5, 8)


def test_full_vocab_kl_direction_and_token_reduction():
    p = torch.tensor([[.8, .2], [.3, .7]]).log()
    q = torch.tensor([[.6, .4], [.4, .6]]).log()
    result = output_metrics(p, q, torch.tensor([0, 1]))
    expected = (.8 * math.log(.8/.6) + .2 * math.log(.2/.4)
                + .3 * math.log(.3/.4) + .7 * math.log(.7/.6)) / 2
    assert result["mean_kl"] == pytest.approx(expected, abs=1e-7)
    assert result["ppl_increase"] == pytest.approx(math.sqrt(.8*.7/(.6*.6)) - 1, abs=1e-7)
    assert output_metrics(p, p, torch.tensor([0, 1]))["mean_kl"] == 0


def test_padding_and_next_token_positions():
    config = AutoMixedPrecisionConfig(nsamples=1, seqlen=8, positions_per_sample=4)
    record = _records([{"input_ids": [0, 3, 4, 5, 0], "attention_mask": [0, 1, 1, 1, 0]}], config)[0]
    assert record[2].tolist() == [1, 2]
    assert record[3].tolist() == [4, 5]
    with pytest.raises(ValueError, match="usable"):
        _records([{"input_ids": [0, 0], "attention_mask": [0, 0]}], config)
    with pytest.raises(ValueError, match="one token sequence"):
        _records([{"input_ids": [[3, 4], [5, 6]]}], config)
    with pytest.raises(ValueError, match="text samples only"):
        _records([{"input_ids": [3, 4], "pixel_values": torch.zeros(1)}], config)


def test_greedy_checks_cumulative_error_and_returns_feasible_plan():
    config = AutoMixedPrecisionConfig(max_kl=.2, max_ppl_increase=.1, max_trials=8)
    # Independent local rankings cannot predict this interacting global objective.
    def evaluate(plan):
        total = sum({4: .4, 5: .15, 8: 0}[b] for b in plan.values())
        return {"mean_kl": total, "ppl_increase": 0}
    result = select_assignment({"0:a": 10, "1:a": 20},
        {u: {4: .4, 5: .15, 8: 0} for u in ("0:a", "1:a")}, evaluate, config)
    assert result["status"] == "accepted"
    assert sorted(result["unit_bits"].values()) == [5, 8]
    assert not result["trials"][0]["accepted"]
    assert any(not t["accepted"] for t in result["trials"][2:])
    failed = select_assignment({"0:a": 10}, {"0:a": {4: 1, 5: .2, 8: 0}},
                               lambda _: {"mean_kl": 1, "ppl_increase": 0}, config)
    assert failed["status"] == "no_feasible_assignment" and failed["unit_bits"] is None
    tight = AutoMixedPrecisionConfig(max_kl=.01, max_trials=2)
    recovered = select_assignment({"0:a": 10, "1:a": 20},
        {u: {4: .4, 5: .15, 8: 0} for u in ("0:a", "1:a")}, evaluate, tight)
    assert recovered["status"] == "accepted"
    assert set(recovered["unit_bits"].values()) == {8}
    assert len(recovered["trials"]) == 2


@pytest.mark.usefixtures("native_qwen35_cpu")
def test_real_hybrid_moe_search_restores_every_weight_and_ties():
    model = tiny_model()
    original = {n: t.clone() for n, t in model.state_dict().items()}
    original_training = [m.training for m in model.modules()]
    assert len(discover_units(model)) == 4
    config = dict(nsamples=2, seqlen=32, positions_per_sample=8, max_trials=3)
    report = search_precision(model, samples(), config, device="cpu")
    assert report["status"] == "accepted"
    assert len(report["module_bits"]) == 13
    assert all(".experts." not in n and "lm_head" not in n and "in_proj_a" not in n
               for n in report["module_bits"])
    assert any(b == 5 for b in report["module_bits"].values())
    assert report["logical_bytes_saved_vs_w8"] > 0
    assert report["trials"][0]["metrics"]["tokens"] == 16
    assert [m.training for m in model.modules()] == original_training
    assert model.lm_head.weight is model.get_input_embeddings().weight
    for n, t in model.state_dict().items():
        torch.testing.assert_close(t, original[n], rtol=0, atol=0)
    with pytest.raises(ValueError, match="complete positive-size"):
        search_precision(model, samples(), config, group_size=65, device="cpu")
    for n, t in model.state_dict().items():
        torch.testing.assert_close(t, original[n], rtol=0, atol=0)
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_batch_refinement_recovers_safe_groups_within_budget():
    units = {str(i): 100 for i in range(8)}
    errors = {u: {4: 2, 5: 1, 8: 0} for u in units}
    calls = []
    def evaluate(plan):
        calls.append(dict(plan))
        return {"mean_kl": 0 if plan["7"] == 8 else .5, "ppl_increase": 0}
    result = select_assignment(units, errors, evaluate,
                               AutoMixedPrecisionConfig(max_kl=.1, max_trials=8))
    assert result["status"] == "accepted"
    assert result["unit_bits"] == {**{str(i): 5 for i in range(7)}, "7": 8}
    assert len(calls) == 8 and result["selected_metrics"]["mean_kl"] == 0
    assert len({tuple(sorted(p.items())) for p in calls}) == len(calls)


@pytest.mark.parametrize("backend", ["gptq", "autoround"])
def test_public_recipe_passes_strict_search_options_to_real_cli(backend, tmp_path):
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround
    recipe = quantize_qwen35 if backend == "gptq" else quantize_qwen35_autoround
    kwargs = dict(model_dir=str(tmp_path / "source"), output_dir=str(tmp_path / "out"),
                  topology="moe", dry_run=True, auto_mixed_precision={"max_kl": .01})
    command = recipe(**kwargs).provenance["command"]
    config = json.loads(command[command.index("--auto-mixed-precision") + 1])
    assert config["max_kl"] == .01 and config["candidates"] == [4, 5, 8]
    assert "--auto-mixed-precision" not in recipe(**dict(kwargs, auto_mixed_precision=False)).provenance["command"]
    with pytest.raises(ValueError, match="MoE"):
        recipe(**dict(kwargs, topology="dense"))
    with pytest.raises(ValueError, match="Unknown"):
        recipe(**dict(kwargs, auto_mixed_precision={"experts": True}))
    with pytest.raises(ValueError, match="overwrite"):
        recipe(**dict(kwargs, output_dir=kwargs["model_dir"]))
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"quantization_config": {"bits": 4}}))
    with pytest.raises(ValueError, match="floating input"):
        recipe(**kwargs)


def test_exact_dynamic_rules_precede_w8_patterns():
    from gptqmodel.quantization.config import dynamic_get
    name = "model.language_model.layers.0.self_attn.q_proj"
    rules = dynamic_overrides({"module_bits": {name: 5}}, {r".*self_attn.*": {"bits": 8},
                              r".*experts.*": {"bits": 4}})
    assert dynamic_get(rules, name, "bits") == 5
    assert dynamic_get(rules, "model.language_model.layers.1.self_attn.q_proj", "bits") == 8
    assert dynamic_get(rules, "model.language_model.layers.0.mlp.experts.0.up_proj", "bits") == 4
