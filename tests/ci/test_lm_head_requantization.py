"""Head-only calibration, producer selection and source-preserving save tests."""
import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import FORMAT
from gptqmodel.quantization.lm_head import HeadQuantizationConfig, resolve_head_algorithm
from gptqmodel.utils.backend import BACKEND


def _tensors(path):
    index = path / "model.safetensors.index.json"
    files = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    result = {}
    for filename in files:
        result.update(load_file(path / filename))
    return result


@pytest.fixture(scope="module", params=[(FORMAT.GPTQ_V2, torch.int32, False), (FORMAT.GPTQ, torch.int32, False),
                                                   (FORMAT.GPTQ_V2, torch.int32, True)])
def checkpoint(tmp_path_factory, request):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("lm-head")
    fp = root / "fp"
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=request.param[2])
    Qwen2ForCausalLM(cfg).save_pretrained(fp)
    # A native tokenizer survives AutoTokenizer's architecture-based dispatch
    # in Transformers 5; a generic WordLevel fixture is replaced by Qwen BPE.
    Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "t": 1, "Ġ": 2,
                          **{str(i): i + 3 for i in range(10)}}, merges=[]).save_pretrained(fp)
    calibration = [{"input_ids": torch.randint(1, 128, (1, 32)), "attention_mask": torch.ones(1, 32, dtype=torch.long)}
                   for _ in range(4)]
    qcfg = QuantizeConfig(bits=4, group_size=64, sym=True, desc_act=False,
                         act_group_aware=False, format=request.param[0], pack_dtype=request.param[1], device="cpu",
                         offload_to_disk=False)
    model = GPTQModel.load(str(fp), qcfg)
    model.quantize(calibration, batch_size=1)
    quant = root / "quant"
    model.save(str(quant))
    return fp, quant, calibration


@pytest.mark.parametrize("bits,algorithm", [(4, "gptq"), (5, "gptq"), (4, "autoround"), (5, "autoround")])
def test_head_only_round_trip(checkpoint, tmp_path, bits, algorithm):
    _, source, calibration = checkpoint
    before = _tensors(source)
    model = GPTQModel.load(str(source), device="cpu", backend=BACKEND.TORCH)
    model.quantize_config.device = "cpu"
    model.requantize(calibration, head={"bits": bits, "algorithm": algorithm, "iters": 2,
                                       "row_chunk_size": 64, "batch_tokens": 32})
    # Reusing an unsaved calibrated head must not discard it in favour of the
    # still-floating head in the source checkpoint when save() follows.
    assert model.requantize(None, head={"bits": bits})["lm_head"][0]["action"] == "skipped"
    target = tmp_path / "result"
    model.save(str(target))
    assert json.loads((target / "lm_head_operation.json").read_text())["action"] == "skipped"
    after = _tensors(target)
    for name, tensor in before.items():
        if not name.startswith("lm_head."):
            assert torch.equal(tensor, after[name]), name
    assert "lm_head.weight" not in after
    word_bits = after["lm_head.qweight"].element_size() * 8
    assert after["lm_head.qweight"].shape == (64 // word_bits * (8 if bits == 5 else bits), 128)
    reloaded = GPTQModel.load(str(target), device="cpu", backend=BACKEND.TORCH)
    assert reloaded.model.lm_head.maxq == 2**bits - 1
    assert not reloaded.model.config.tie_word_embeddings
    if model.model.lm_head.qzero_format() == 1:
        from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module
        convert_gptq_v1_to_v2_format_module(model.model.lm_head,
            bits=model.model.lm_head.bits, pack_dtype=model.model.lm_head.pack_dtype)
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        actual = getattr(reloaded.model.lm_head, name)
        # The CPU loader casts floating buffers to its BF16 inference dtype.
        expected = getattr(model.model.lm_head, name)
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
    # Compare the head as serialized, independent of inference-time qzero flags.
    x = torch.randn(2, 64, dtype=reloaded.model.lm_head.scales.dtype)
    assert torch.isfinite(reloaded.model.lm_head(x)).all()
    config = json.loads((target / "config.json").read_text())["quantization_config"]
    assert config["meta"]["lm_head_quantization"]["algorithm"] == algorithm
    original_config = reloaded.quantize_config
    # Algorithm options apply only to calibration; a skip keeps the producer.
    operation = reloaded.requantize(None, head={"bits": bits,
        "algorithm": "autoround" if algorithm == "gptq" else "gptq"})["lm_head"][0]
    assert operation["action"] == "skipped"
    assert operation["actual"]["algorithm"] == algorithm
    assert reloaded.quantize_config is original_config
    reused = tmp_path / "reused"
    reloaded.save(str(reused))
    for filename in target.glob("*.safetensors"):
        assert filename.read_bytes() == (reused / filename.name).read_bytes()
    assert (reused / "config.json").read_bytes() == (target / "config.json").read_bytes()
    assert json.loads((reused / "lm_head_operation.json").read_text())["action"] == "skipped"
    with pytest.raises(ValueError, match="already packed"):
        reloaded.requantize(None, head={"bits": 8})

    # The recipe skips before reading any calibration data or replaying the body.
    from gptqmodel.recipes.qwen35_head import quantize_qwen35_head
    recipe_target = tmp_path / "recipe-reused"
    result = quantize_qwen35_head(model_dir=str(target), output_dir=str(recipe_target),
                                 head={"bits": bits}, device="cpu")
    assert result.method == "skip"
    assert result.provenance["action"] == "skipped"
    assert result.output_dir == str(target)
    assert not recipe_target.exists()
    recipe_tensors = _tensors(target)
    for name, tensor in after.items():
        assert torch.equal(tensor, recipe_tensors[name]), name


def test_packed_head_requires_explicit_float_source(checkpoint, tmp_path):
    fp, source, calibration = checkpoint
    model = GPTQModel.load(str(source), device="cpu", backend=BACKEND.TORCH)
    with pytest.raises(ValueError, match="requires calibration data"):
        model.requantize(head={"bits": 8})
    with pytest.raises(ValueError, match="requantize"):
        model.quantize(calibration, lm_head_only=True)
    model.requantize(calibration, head={"bits": 8})
    model.save(str(tmp_path / "head8"))
    model = GPTQModel.load(str(tmp_path / "head8"), device="cpu", backend=BACKEND.TORCH)
    assert model.requantize(head={"bits": 8})["lm_head"][0]["action"] == "skipped"
    with pytest.raises(ValueError, match="cannot be skipped"):
        model.requantize(None, head={"bits": 5})
    model.requantize(calibration, head={"bits": 5, "source_model_dir": str(fp)})
    model.save(str(tmp_path / "head5"))
    reloaded = GPTQModel.load(str(tmp_path / "head5"), device="cpu", backend=BACKEND.TORCH)
    assert reloaded.model.lm_head.maxq == 31
    assert "lm_head.weight" not in _tensors(tmp_path / "head5")


def test_compatibility_recipe_saves_sibling_and_reuses_it(checkpoint, tmp_path):
    from gptqmodel.recipes.qwen35_head import quantize_qwen35_head

    _, source, _ = checkpoint
    before = _tensors(source)
    data = tmp_path / "calibration.jsonl"
    data.write_text("\n".join(json.dumps({"text": " ".join(f"t{i}" for i in range(1, 33))}) for _ in range(4)))
    target = source.with_name(source.name + "_head_w4")
    with pytest.warns(UserWarning, match="floating head") as caught:
        result = quantize_qwen35_head(model_dir=str(source), head={"bits": 4},
            calibration_jsonl=str(data), nsamples=4, seqlen=32, device="cpu")
    assert str(target) in str(caught[0].message)
    assert result.output_dir == str(target)
    assert result.provenance["action"] == "calibrated"
    after = _tensors(target)
    unchanged = _tensors(source)
    for name, value in before.items():
        assert torch.equal(value, unchanged[name])
        if not name.startswith("lm_head."):
            assert torch.equal(value, after[name])
    reused = quantize_qwen35_head(model_dir=result.output_dir, head={"bits": 4}, device="cpu")
    assert reused.output_dir == result.output_dir
    assert reused.method == "skip"
    with pytest.raises(FileExistsError, match="Pass this checkpoint"):
        quantize_qwen35_head(model_dir=str(source), head={"bits": 4}, device="cpu")


def test_recipe_w8_keeps_legacy_path_without_reading_checkpoint(tmp_path):
    from gptqmodel.recipes.qwen35_head import quantize_qwen35_head
    source = tmp_path / "unused-input"
    result = quantize_qwen35_head(model_dir=str(source), head={"bits": 8})
    assert result.output_dir == str(source)
    assert result.method == "skip"
    assert not source.exists()


def test_full_recipe_publishes_only_final_checkpoint(checkpoint, tmp_path):
    import shutil
    from gptqmodel.recipes.qwen35 import Qwen35QuantRecipeResult
    from gptqmodel.recipes.qwen35_head import quantize_with_head

    fp, quantized, _ = checkpoint
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": "t1 t2 " * 32}) for _ in range(4)))

    def backbone_recipe(**kwargs):
        # Substitute only the expensive full-body recipe boundary; the fixture
        # was genuinely quantized. Head calibration and publication run for real.
        output = kwargs["output_dir"]
        log = output + ".log"
        if not kwargs["dry_run"]:
            shutil.copytree(quantized, output)
            from pathlib import Path
            Path(log).write_text("backbone calibration log\n")
        return Qwen35QuantRecipeResult(output, "gptq", "gptqmodel_hf", "qwen35", "dense",
                                      "llm_only", None, False, False, {"log_file": log})

    target = tmp_path / "final"
    result = quantize_with_head(backbone_recipe, dict(model_dir=str(fp), output_dir=str(target),
        head={"bits": 4}, calibration_jsonl=str(data), nsamples=4, seqlen=32,
        device="cpu", dry_run=False, seed=42), backbone_algorithm="gptq")
    assert result.output_dir == str(target)
    assert not list(tmp_path.glob("final.backbone*"))
    assert (target / "backbone_quantization.log").read_text() == "backbone calibration log\n"
    assert result.provenance["log_file"] == str(target / "backbone_quantization.log")
    assert result.provenance["head_stage"]["action"] == "calibrated"
    assert "lm_head.qweight" in _tensors(target)


def test_native_qwen35_tied_head_is_independent_after_save(tmp_path):
    from types import SimpleNamespace
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    from gptqmodel.quantization.lm_head import untie_float_head

    config = Qwen3_5MoeConfig(tie_word_embeddings=True, text_config={
        "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "layer_types": ["full_attention"], "num_experts": 2,
        "num_experts_per_tok": 1, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64, "tie_word_embeddings": True,
    }, vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
        "num_heads": 4, "out_hidden_size": 64, "num_position_embeddings": 16})
    model = Qwen3_5MoeForConditionalGeneration(config)
    embedding = model.get_input_embeddings().weight
    assert model.lm_head.weight is embedding
    before = embedding.detach().clone()
    owner = SimpleNamespace(model=model, lm_head="lm_head")
    untie_float_head(owner)
    assert model.lm_head.weight.data_ptr() != embedding.data_ptr()
    model.lm_head.weight.data.add_(1)
    torch.testing.assert_close(embedding, before, rtol=0, atol=0)
    model.save_pretrained(tmp_path / "untied")
    reloaded = Qwen3_5MoeForConditionalGeneration.from_pretrained(tmp_path / "untied")
    assert not reloaded.config.tie_word_embeddings
    assert not reloaded.config.text_config.tie_word_embeddings
    torch.testing.assert_close(reloaded.get_input_embeddings().weight, before, rtol=0, atol=0)
    torch.testing.assert_close(reloaded.lm_head.weight, before + 1, rtol=0, atol=0)


def test_autoround_storage_tag_does_not_select_gptq(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "gptq", "checkpoint_format": "gptq_v2", "provider": "auto-round"}}))
    assert resolve_head_algorithm(tmp_path) == "autoround"
    assert resolve_head_algorithm(tmp_path, "gptq") == "gptq"


def test_unknown_producer_requires_explicit_choice(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="infer"):
        resolve_head_algorithm(tmp_path)
    assert resolve_head_algorithm(tmp_path, "autoround") == "autoround"
    with pytest.raises(ValueError, match="bits"):
        HeadQuantizationConfig(bits=3)


@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int64])
def test_unsupported_word_dtype_fails_before_restoring_head(tmp_path, pack_dtype):
    from types import SimpleNamespace
    from gptqmodel.quantization.lm_head import prepare_head_quantization

    (tmp_path / "config.json").write_text('{"quantization_config":{"quant_method":"gptq"}}')
    model = SimpleNamespace(model_local_path=str(tmp_path),
                            quantize_config=SimpleNamespace(pack_dtype=pack_dtype))
    with pytest.raises(ValueError, match="pack_dtype=torch.int32"):
        prepare_head_quantization(model, {"bits": 5, "source_model_dir": str(tmp_path / "missing-fp")})


def test_sidecar_only_checkpoint_retains_producer(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "quantize_config.json").write_text(json.dumps({
        "quant_method": "gptq", "meta": {"auto_round": {"auto_round_enabled": True}}}))
    assert resolve_head_algorithm(tmp_path) == "autoround"


@pytest.mark.parametrize("change", ["logical_bits", "group_size", "sym", "desc_act", "pack_dtype",
                                    "g_idx", "g_idx_dtype", "qzeros", "data_type"])
def test_skip_rejects_incompatible_packed_layout(tmp_path, change):
    from types import SimpleNamespace
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.quantization.lm_head import head_skip_info

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "quant_method": "gptq", "data_type": "int_sym_xh"}}))
    packed = TorchQuantLinear(bits=5, group_size=64, sym=True, desc_act=False,
                               in_features=128, out_features=64, bias=False)
    linear = torch.nn.Linear(128, 64, bias=False)
    linear.weight.data.copy_((torch.arange(128 * 64).reshape(64, 128) % 32 - 16).float())
    packed.pack_original(linear, torch.ones(64, 2), torch.full((64, 2), 16.),
                         torch.arange(128, dtype=torch.int32) // 64)
    packed.qzero_format(format=2)
    module = torch.nn.Module()
    module.lm_head = packed
    owner = SimpleNamespace(model=module, lm_head="lm_head", model_local_path=str(tmp_path),
        quantize_config=QuantizeConfig(bits=4, group_size=64, format=FORMAT.GPTQ_V2))
    request = {"bits": 5}
    assert head_skip_info(owner, request)["action"] == "skipped"
    # A float source explicitly requests recalibration, even at identical bits.
    assert head_skip_info(owner, {**request, "source_model_dir": str(tmp_path / "fp")}) is None
    if change == "g_idx":
        packed.g_idx.zero_()
    elif change == "g_idx_dtype":
        packed.g_idx = packed.g_idx.float()
    elif change == "qzeros":
        packed.qzeros = packed.qzeros[:, :-1]
    elif change == "data_type":
        request["data_type"] = "int_sym_gptq"
    else:
        setattr(packed, change, {"logical_bits": 8, "group_size": 32, "sym": False,
                                "desc_act": True, "pack_dtype": torch.int16}[change])
    with pytest.raises(ValueError, match="cannot be skipped"):
        head_skip_info(owner, request)


@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_full_recipe_inherits_head_algorithm(algorithm, tmp_path):
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    source = tmp_path / "fp"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    data = tmp_path / "calib.jsonl"
    data.write_text('{"text":"t1 t2"}\n')
    options = dict(model_dir=str(source), output_dir=str(tmp_path / "out"),
                   topology="moe", head={"bits": 5}, nsamples=8, seqlen=32, seed=13, dry_run=True)
    if algorithm == "gptq":
        result = quantize_qwen35(**options, calibration_jsonl=str(data))
    else:
        result = quantize_qwen35_autoround(**options, dataset=str(data))
    assert result.provenance["head_stage"]["head"]["algorithm"] == algorithm
    assert result.provenance["head_stage"]["head"]["seed"] == 13
    assert result.output_dir == str(tmp_path / "out")
    assert not (tmp_path / "out.backbone").exists()
    if algorithm == "gptq":
        with pytest.raises(ValueError, match="rotation"):
            quantize_qwen35(**options, calibration_jsonl=str(data), rotation="hadamard")
    recipe = quantize_qwen35 if algorithm == "gptq" else quantize_qwen35_autoround
    data_options = {"calibration_jsonl": str(data)} if algorithm == "gptq" else {"dataset": str(data)}
    baseline = recipe(**dict(options, head=None), **data_options)
    w8 = recipe(**dict(options, head={"bits": 8}), **data_options)
    assert w8.provenance == baseline.provenance
    with pytest.raises(ValueError, match="head.bits"):
        recipe(**dict(options, head={"bits": 3}), **data_options)


def test_autoround_head_accepts_default_hf_dataset(tmp_path):
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    result = quantize_qwen35_autoround(model_dir=str(source), output_dir=str(tmp_path / "out") + "/",
        topology="moe", head={"bits": 5}, dry_run=True)
    stage = result.provenance["head_stage"]
    assert stage["calibration_dataset"] == "NeelNanda/pile-10k"
    assert stage["calibration_jsonl"] is None
    assert "--calibration-dataset" in stage["command"]
    assert result.output_dir == str(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_qwen35_reused_dataset_method_accepts_tokenized_text():
    from types import SimpleNamespace
    from gptqmodel.models.definitions.qwen3_5_moe import Qwen3_5_MoeQModel
    from gptqmodel.models.base import BaseQModel

    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    owner = SimpleNamespace(tokenizer=tokenizer)
    inputs = [{"input_ids": torch.arange(1, 33)[None], "attention_mask": torch.ones(1, 32)}]
    expected = BaseQModel.prepare_dataset(owner, inputs, batch_size=1)
    actual = Qwen3_5_MoeQModel.prepare_dataset(owner, inputs, batch_size=1)
    assert len(actual) == len(expected)
    for key in ("input_ids", "attention_mask"):
        torch.testing.assert_close(actual[0][key], expected[0][key])
