"""Native vision/M-RoPE, answer masking and independent image search gates."""
import json

import pytest
import torch
import torch.nn.functional as F

from _qwen35_mixed_fixture import native_qwen35_cpu, samples, save_image_dataset, save_source, tiny_model
from gptqmodel.quantization.auto_mixed_precision import (
    aggregate_metrics, discover_units, load_image_records, metrics_pass, output_metrics,
    parse_auto_mixed_precision, search_precision, select_assignment, symmetric_qdq,
)


@pytest.mark.parametrize("value", [{"image_dataset": ""}, {"image_dataset": []},
    {"image_nsamples": 0}, {"image_nsamples": True}, {"image_seqlen": 1}, {"image_max_pixels": -1}])
def test_invalid_image_config(value):
    with pytest.raises(ValueError):
        parse_auto_mixed_precision(value)


def test_image_failure_is_not_hidden_by_text_average():
    config = parse_auto_mixed_precision({"image_dataset": "images.jsonl", "max_trials": 3, "max_kl": .015})
    def evaluate(plan):
        image_kl = .02 if plan["0:a"] < 8 else .001
        return {"mean_kl": .001, "ppl_increase": 0, "modalities": {
            "text": {"mean_kl": 0, "ppl_increase": 0},
            "image": {"mean_kl": image_kl, "ppl_increase": 0}}}
    assert not metrics_pass(evaluate({"0:a": 5}), config)
    assert not metrics_pass({"mean_kl": 0, "ppl_increase": 0}, config)
    result = select_assignment({"0:a": 10}, {"0:a": {4: 2, 5: 1, 8: 0}}, evaluate, config)
    assert result["unit_bits"] == {"0:a": 8}
    metric = evaluate({"0:a": 8})
    metric["modalities"]["image"]["ppl_increase"] = .03
    assert not metrics_pass(metric, config)


@pytest.mark.usefixtures("native_qwen35_cpu")
def test_real_vision_search_cache_matches_uncached_forward(tmp_path):
    source = save_source(tmp_path / "source")
    dataset = save_image_dataset(tmp_path / "images")
    options = dict(image_dataset=str(dataset), image_nsamples=2, image_max_pixels=4096,
                   image_seqlen=128, nsamples=2, seqlen=32, positions_per_sample=8,
                   max_trials=2, max_kl=1, max_ppl_increase=1)
    config = parse_auto_mixed_precision(options)
    images = load_image_records(source, config)
    assert all(r["labels"].numel() == 1 for r in images)
    assert all(r["inputs"]["mm_token_type_ids"][0, r["positions"]].eq(0).all() for r in images)
    model = tiny_model().eval()
    before = {n: t.clone() for n, t in model.state_dict().items()}
    model.model.rope_deltas = torch.tensor([[7]])
    vision_calls = []
    hook = model.model.visual.register_forward_hook(lambda *args: vision_calls.append(1))
    try:
        report = search_precision(model, samples(), options, image_samples=images, device="cpu")
    finally:
        hook.remove()
    assert len(vision_calls) == 2  # Each image encoded once, even across candidate/control trials.
    assert report["modality_sample_counts"] == {"text": 2, "image": 2}
    assert report["status"] == "accepted" and metrics_pass(report["selected_metrics"], config)
    assert report["all_w8_metrics"]["modalities"]["image"]["choices"]["count"] == 2
    assert model.model.rope_deltas.item() == 7
    assert all("cached_inputs" not in image for image in images)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert all(not m._forward_pre_hooks for m in model.modules())

    def forward(record):
        return F.log_softmax(model(**record["inputs"], use_cache=False,
                                  logits_to_keep=record["positions"]).logits[0].float(), -1)
    with torch.inference_mode():
        reference = [forward(r) for r in images]
        for group in discover_units(model).values():
            for module in group.values():
                module.weight.copy_(symmetric_qdq(module.weight, 8, 64))
        actual = aggregate_metrics([output_metrics(ref, forward(r), r["labels"])
                                    for ref, r in zip(reference, images)])
    cached = report["all_w8_metrics"]["modalities"]["image"]
    assert cached["mean_kl"] == pytest.approx(actual["mean_kl"], abs=1e-7)
    assert cached["delta_nll"] == pytest.approx(actual["delta_nll"], abs=1e-7)

    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    rows[0].pop("options")
    rows[0]["answer"] = "t7 t8 t9"
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    open_answers = load_image_records(source, config)
    answer = next(r for r in open_answers if r["labels"].numel() == 3)
    torch.testing.assert_close(answer["labels"], torch.tensor([7, 8, 9]))
    torch.testing.assert_close(answer["inputs"]["input_ids"][0, answer["positions"] + 1], answer["labels"])
    assert answer["inputs"]["mm_token_type_ids"][0, answer["positions"] + 1].eq(0).all()
    with pytest.raises(ValueError, match="do not truncate images"):
        load_image_records(source, parse_auto_mixed_precision({**options, "image_seqlen": 2}))
    rows[0]["answer"] = ""
    dataset.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="nonempty"):
        load_image_records(source, config)
