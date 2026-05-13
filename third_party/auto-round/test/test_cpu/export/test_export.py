import json
import os
import shutil

import pytest
import torch
from packaging import version
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.export.export_to_autoround import export as autoround_export

from ...helpers import get_model_path, opt_name_or_path, transformers_version


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.model_name = opt_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_autogptq_format(self, dataloader):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=dataloader,
            )

            quantized_model_path = self.save_dir
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")

            if group_size == -1:
                continue
            quantization_config = AutoRoundConfig()
            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="auto", trust_remote_code=True, quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_format(self, dataloader):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, True
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=dataloader,
            )
            quantized_model_path = self.save_dir
            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

            if group_size == -1:
                continue
            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_gptqmodel_format_with_five_bit_override(self, dataloader):
        from safetensors import safe_open

        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model=self.model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config={
                "model.decoder.layers.0.self_attn.q_proj": {
                    "bits": 5,
                    "group_size": group_size,
                    "sym": sym,
                }
            },
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        with open(os.path.join(quantized_model_path, "quantization_config.json"), "r", encoding="utf-8") as f:
            quant_config = json.load(f)
        assert quant_config["packing_format"] == "auto_round:gptqmodel"
        assert quant_config["quant_method"] == "gptq"
        assert quant_config["provider"] == "auto-round"
        assert quant_config["checkpoint_format"] == "gptq_v2"
        assert quant_config["dynamic"]["+:^model\\.decoder\\.layers\\.0\\.self_attn\\.q_proj$"]["bits"] == 5

        qweight_shapes = {}
        qzeros_shapes = {}
        for filename in os.listdir(quantized_model_path):
            if not filename.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(quantized_model_path, filename), framework="pt", device="cpu") as tensors:
                for key in (
                    "model.decoder.layers.0.self_attn.q_proj.qweight",
                    "model.decoder.layers.0.self_attn.q_proj.qzeros",
                    "model.decoder.layers.0.self_attn.k_proj.qweight",
                    "model.decoder.layers.0.self_attn.k_proj.qzeros",
                ):
                    if key in tensors.keys():
                        target = qweight_shapes if key.endswith("qweight") else qzeros_shapes
                        target[key] = tuple(tensors.get_tensor(key).shape)

        assert qweight_shapes["model.decoder.layers.0.self_attn.q_proj.qweight"] == (768 // 32 * 8, 768)
        assert qzeros_shapes["model.decoder.layers.0.self_attn.q_proj.qzeros"] == (768 // group_size, 768 // 32 * 8)
        assert qweight_shapes["model.decoder.layers.0.self_attn.k_proj.qweight"] == (768 // 32 * 4, 768)
        assert qzeros_shapes["model.decoder.layers.0.self_attn.k_proj.qzeros"] == (768 // group_size, 768 // 32 * 4)

    def test_autoround_gptqmodel_format_global_five_bit(self, dataloader):
        from safetensors import safe_open

        bits, group_size, sym = 5, 128, True
        autoround = AutoRound(
            model=self.model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        with open(os.path.join(quantized_model_path, "quantization_config.json"), "r", encoding="utf-8") as f:
            quant_config = json.load(f)

        assert quant_config["bits"] == 5
        assert quant_config["packing_format"] == "auto_round:gptqmodel"
        assert quant_config["quant_method"] == "gptq"
        assert quant_config["checkpoint_format"] == "gptq_v2"

        qweight_shape = None
        qzeros_shape = None
        for filename in os.listdir(quantized_model_path):
            if not filename.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(quantized_model_path, filename), framework="pt", device="cpu") as tensors:
                qweight_key = "model.decoder.layers.0.self_attn.q_proj.qweight"
                qzeros_key = "model.decoder.layers.0.self_attn.q_proj.qzeros"
                if qweight_shape is None and qweight_key in tensors.keys():
                    qweight_shape = tuple(tensors.get_tensor(qweight_key).shape)
                if qzeros_shape is None and qzeros_key in tensors.keys():
                    qzeros_shape = tuple(tensors.get_tensor(qzeros_key).shape)

        assert qweight_shape == (768 // 32 * 8, 768)
        assert qzeros_shape == (768 // group_size, 768 // 32 * 8)

    def test_autoround_gptqmodel_format_writes_negative_dynamic_for_fp_layers(self, dataloader):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model=self.model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            layer_config={
                "model.decoder.layers.0.self_attn.q_proj": {
                    "bits": 5,
                    "group_size": group_size,
                    "sym": sym,
                },
                "model.decoder.layers.0.self_attn.k_proj": {
                    "bits": 16,
                    "group_size": group_size,
                    "sym": sym,
                },
            },
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:gptqmodel")

        with open(os.path.join(quantized_model_path, "quantization_config.json"), "r", encoding="utf-8") as f:
            quant_config = json.load(f)

        dynamic = quant_config["dynamic"]
        assert dynamic["+:^model\\.decoder\\.layers\\.0\\.self_attn\\.q_proj$"]["bits"] == 5
        assert "-:^model\\.decoder\\.layers\\.0\\.self_attn\\.k_proj$" in dynamic
        assert "+:^model\\.decoder\\.layers\\.0\\.self_attn\\.k_proj$" not in dynamic

    def test_compact_gptqmodel_dynamic_overrides_drops_exact_rules_covered_by_same_regex(self):
        dynamic = {
            "+:^model\\.layers\\.0\\.mlp\\.experts\\.0\\.down_proj$": {"bits": 5},
            "+:^model\\.layers\\.0\\.mlp\\.experts\\.1\\.down_proj$": {"bits": 5},
            "+:^model\\.layers\\.0\\.self_attn\\.q_proj$": {"bits": 5},
            "+:.*model\\.layers.*\\.mlp\\.experts.*\\.down_proj.*": {"bits": 5},
        }

        compacted = autoround_export._compact_gptqmodel_dynamic_overrides(dynamic)

        assert "+:.*model\\.layers.*\\.mlp\\.experts.*\\.down_proj.*" in compacted
        assert "+:^model\\.layers\\.0\\.mlp\\.experts\\.0\\.down_proj$" not in compacted
        assert "+:^model\\.layers\\.0\\.mlp\\.experts\\.1\\.down_proj$" not in compacted
        assert "+:^model\\.layers\\.0\\.self_attn\\.q_proj$" in compacted

    def test_autoround_awq_format(self, dataloader):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            model_name = self.model_name
            autoround = AutoRound(
                model=model_name,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=dataloader,
            )
            quantized_model_path = self.save_dir

            autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round:auto_awq")

            # quantization_config = AutoRoundConfig(
            #     backend="cpu"
            # )
            if group_size == -1:
                continue

            model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoawq_format(self, dataloader):
        for group_size in [-1, 32, 128]:
            bits, sym = 4, False
            autoround = AutoRound(
                self.model,
                self.tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=dataloader,
            )
            autoround.quantize()
            quantized_model_path = self.save_dir

            autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_awq")
            if group_size == -1:
                continue
            quantization_config = AutoRoundConfig()

            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="cpu", quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_3bit_asym_format(self, dataloader):
        bits, group_size, sym = 3, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        device = "cpu"  ##cpu, hpu, cuda
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_autoround_3bit_sym_format(self, dataloader):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir

        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")
        device = "cpu"  ##cpu, hpu, cuda
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    @pytest.mark.parametrize("static_kv_dtype", ["fp8", "float16"])
    def test_static_afp8_export(self, static_kv_dtype):
        import os

        from safetensors import safe_open

        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=8,
            group_size=-1,
            iters=0,
            scheme="fp8_static",
            nsamples=2,
            seqlen=2,
            static_kv_dtype=static_kv_dtype,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        if static_kv_dtype is None:
            with torch.no_grad():
                import transformers

                model = transformers.AutoModelForCausalLM.from_pretrained(
                    quantized_model_path,
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                model.eval()
                assert (
                    model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__
                    == "WeightFP8ActFP8StaticQuantLinear"
                ), f"Expected WeightFP8ActFP8StaticQuantLinear, got {model.model.decoder.layers[0].self_attn.k_proj.__class__.__name__}"
                tokenizer = transformers.AutoTokenizer.from_pretrained(quantized_model_path)
                prompt = "AI is "
                encode = tokenizer.encode(prompt, return_tensors="pt")
                with torch.no_grad():
                    output_tokens = model.generate(
                        encode,
                        max_length=10,
                    )
                    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                    print(f"Prompt: {prompt}")
                    print(f"Output: {output}")
                    assert output is not None, "Output should not be None"

        if static_kv_dtype == "fp8":
            assert "model.decoder.layers.8.self_attn.k_scale" in f.keys()
            assert "model.decoder.layers.8.self_attn.v_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.5.self_attn.v_scale").shape == torch.Size([1])
            assert f.get_tensor("model.decoder.layers.5.self_attn.k_scale").shape == torch.Size([1])
            assert (
                f.get_tensor("model.decoder.layers.5.self_attn.k_scale").dtype == torch.float32
                or f.get_tensor("model.decoder.layers.5.self_attn.k_scale").dtype == torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=8,
            group_size=-1,
            iters=1,
            act_bits=8,
            nsamples=2,
            seqlen=2,
            data_type="fp8",
            act_data_type="fp8",
            act_dynamic=False,
            act_group_size=0,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")

        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn

    def test_static_fp8_attn(self):
        import os

        from safetensors import safe_open

        model_name = self.model_name
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        autoround = AutoRound(
            model,
            self.tokenizer,
            iters=0,
            nsamples=2,
            seqlen=2,
            scheme="FP8_STATIC",
            static_attention_dtype="fp8",
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        f = safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt")
        assert "model.decoder.layers.8.self_attn.k_proj.input_scale" in f.keys()
        assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.input_scale").shape == torch.Size([1])
        assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.float8_e4m3fn
        check_attrs = ["k_scale", "v_scale", "q_scale"]
        for attr in check_attrs:
            weight_name = f"model.decoder.layers.8.self_attn.{attr}"
            assert weight_name in f.keys()
            assert f.get_tensor(weight_name).shape == torch.Size([1])
            assert f.get_tensor(weight_name).dtype == torch.float32 or f.get_tensor(weight_name).dtype == torch.bfloat16

    def test_awq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, False, 128
        model_name = get_model_path("microsoft/phi-4")
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            nsamples=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_awq")
        lm_head = compressed_model.lm_head
        from auto_round.export.export_to_awq.utils import WQLinear_GEMM

        assert isinstance(lm_head, WQLinear_GEMM), "Illegal AWQ quantization for lm_head layer"

    def test_gptq_lmhead_export(self, dataloader):
        bits, sym, group_size = 4, True, 128
        # Note that, to save UT tuning time, the local model is intentionally kept lightweight, using only 2 hidden layers.
        model_name = get_model_path("microsoft/phi-4")
        layer_config = {
            "lm_head": {"bits": 4},  # set lm_head quant
            "layer": {"bits": 16},
        }
        autoround = AutoRound(
            model=model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            nsamples=2,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_gptq")
        lm_head = compressed_model.lm_head
        assert hasattr(lm_head, "bits") and lm_head.bits == 4, "Illegal GPTQ quantization for lm_head layer"
        quantization_config = AutoRoundConfig()
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)

    def test_export_format(self):
        from auto_round.formats import get_formats

        autoround = AutoRound(
            self.model_name,
            scheme="FP8_STATIC",
        )
        format_list = get_formats("auto_round, llm_compressor, auto_round:llm_compressor", autoround)
        assert len(format_list) == 3
        assert format_list[0].output_format == "auto_round"
        assert format_list[0].get_backend_name() == "auto_round:fp8_static"
        assert format_list[1].output_format == "llm_compressor"
        assert format_list[1].get_backend_name() == "llm_compressor:fp8_static"
        assert format_list[2].output_format == "auto_round"
        assert format_list[2].get_backend_name() == "auto_round:llm_compressor:fp8_static"

        autoround = AutoRound(
            self.model_name,
            scheme="W4A16",
        )
        format_list = get_formats("auto_round:auto_awq, auto_gptq", autoround)
        assert format_list[0].output_format == "auto_round"
        assert format_list[0].get_backend_name() == "auto_round:auto_awq"
        assert format_list[1].output_format == "auto_gptq"
        assert format_list[1].get_backend_name() == "auto_gptq"

        autoround = AutoRound(
            model=self.model_name,
            scheme="INT8_W8A8",
        )
        format_list = get_formats("llm_compressor, auto_round:llm_compressor", autoround)
        assert format_list[0].output_format == "llm_compressor"
        assert format_list[0].get_backend_name() == "llm_compressor:int8_w8a8"
        assert format_list[1].output_format == "auto_round"
        assert format_list[1].get_backend_name() == "auto_round:llm_compressor:int8_w8a8"

    def test_export_format_with_scheme(self, tiny_qwen_model_path):
        from auto_round.formats import get_formats

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="W4A16",
            bits=2,
            group_size=32,
            sym=True,
        )
        with pytest.raises(ValueError, match="auto_awq format support quantization scheme with W4A16 but got bits=2"):
            get_formats("auto_round:auto_awq", ar)

        with pytest.raises(ValueError, match="but got bits=2, data_type=int"):
            get_formats("auto_round:llm_compressor", ar)

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="FP8_STATIC",
            bits=4,
            group_size=32,
            sym=True,
        )
        with pytest.raises(ValueError, match="but got data_type=fp, bits=4"):
            get_formats("auto_round:llm_compressor", ar)

        ar = AutoRound(
            model=tiny_qwen_model_path,
            scheme="w2a16",
            bits=4,
            group_size=256,
            sym=True,
        )
        get_formats("auto_round:auto_awq", ar)

    def test_autoawq_qwen3_vl_infer(self, dataloader):
        model_path = get_model_path("Qwen/Qwen3-VL-2B-Instruct")
        autoround = AutoRound(
            model=model_path,
            scheme="W4A16",
            iters=0,
            seqlen=2,
            batch_size=1,
            dataset=dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=False, format="auto_awq")

        # Check items of modules_to_not_convert in quantization config
        quantization_config_path = f"{quantized_model_path}/quantization_config.json"
        with open(quantization_config_path, "r") as f:
            quantization_config = json.load(f)
        modules_to_not_convert = quantization_config.get("modules_to_not_convert", [])
        assert (
            "model.visual.merger.linear_fc2" in modules_to_not_convert
        ), f"'model.visual.merger.linear_fc2' should be in modules_to_not_convert. Got: {modules_to_not_convert}"
        assert (
            "model.visual.merger.linear_fc1" in modules_to_not_convert
        ), f"'model.visual.merger.linear_fc1' should be in modules_to_not_convert. Got: {modules_to_not_convert}"
        assert (
            "model.visual.blocks" in modules_to_not_convert
        ), f"'model.visual.blocks' should be in modules_to_not_convert. Got: {modules_to_not_convert}"

    def test_llmc_dynamic_wint8aint8_export(self):
        from safetensors import safe_open

        autoround = AutoRound(
            self.model_name,
            iters=0,
            nsamples=2,
            seqlen=2,
            scheme="INT8_W8A8",
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.int8

    def test_llmc_dynamic_wint8aint8_export_with_tuning(self, dataloader):
        from safetensors import safe_open

        autoround = AutoRound(self.model_name, iters=1, nsamples=2, seqlen=2, dataset=dataloader, scheme="INT8_W8A8")
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        with safe_open(os.path.join(quantized_model_path, "model.safetensors"), framework="pt") as f:
            assert "model.decoder.layers.8.self_attn.k_proj.weight_scale" in f.keys()
            assert f.get_tensor("model.decoder.layers.5.self_attn.v_proj.weight").dtype == torch.int8
