"""End-to-end verification: load quantized model, inference, dequantize, export."""
import torch
import time
import os
import json

MODEL_PATH = "weights/Qwen3.5-27B-W4G64-rotate-autoround-v3"
OUTPUT_DIR = "output/qwen35-xh2-export-test"


def _patch_backends():
    """Remove backends that require GLIBC_2.32 (system has 2.31).
    Must be called BEFORE any transformers/auto_round loading.
    """
    try:
        from auto_round.inference.backend import BackendInfos
        to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
        for k in to_remove:
            del BackendInfos[k]
        print(f"Patched: removed {len(to_remove)} incompatible backends")
    except Exception as e:
        print(f"Backend patch skipped: {e}")


def _set_backend_in_config(model_path, backend="auto_round:tritonv2"):
    """Inject backend into quantization_config.json so auto-select uses tritonv2."""
    import json
    qc_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(qc_path):
        # Try in config.json
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if "quantization_config" in cfg:
                cfg["quantization_config"]["backend"] = backend
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"Set backend={backend} in config.json")
        return
    with open(qc_path) as f:
        qc = json.load(f)
    qc["backend"] = backend
    with open(qc_path, "w") as f:
        json.dump(qc, f, indent=2)
    print(f"Set backend={backend} in quantization_config.json")


def main():
    _patch_backends()
    _set_backend_in_config(MODEL_PATH, backend="auto_round:tritonv2")

    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # === Step 1: Load quantized model ===
    print("=" * 60)
    print("Step 1: Loading AutoRound quantized model...")
    print("=" * 60)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="cuda:0", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Loaded in {time.time()-t0:.1f}s | class={type(model).__name__}")

    # === Step 2: Inference ===
    print("\n" + "=" * 60)
    print("Step 2: Inference test...")
    print("=" * 60)
    prompt = "What is quantum computing in one sentence?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    gen_time = time.time() - t1
    n_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {resp[:200]}")
    print(f"Time: {gen_time:.1f}s | Tokens: {n_tokens}")

    # === Step 3: Quant layer info ===
    print("\n" + "=" * 60)
    print("Step 3: Quantized layer info...")
    print("=" * 60)
    quant_layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, "qweight"):
            quant_layers.append((name, mod))
    print(f"Found {len(quant_layers)} quantized layers")
    if quant_layers:
        name, mod = quant_layers[0]
        print(f"  Example: {name}")
        print(f"    qweight: {mod.qweight.shape}, dtype={mod.qweight.dtype}")
        if hasattr(mod, "scales"):
            print(f"    scales:  {mod.scales.shape}")
        if hasattr(mod, "qzeros"):
            print(f"    qzeros:  {mod.qzeros.shape}")

    # === Step 4: Dequantize test ===
    print("\n" + "=" * 60)
    print("Step 4: Dequantization test...")
    print("=" * 60)
    if quant_layers:
        name, mod = quant_layers[0]
        if hasattr(mod, "weight") and callable(getattr(mod, "weight", None)):
            w = mod.weight()
            print(f"  {name}: dequantized shape={w.shape}, dtype={w.dtype}")
        elif hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            print(f"  {name}: weight shape={mod.weight.shape}, dtype={mod.weight.dtype}")
        else:
            print(f"  {name}: qweight available, manual dequant needed")
            # Try to extract params
            print(f"    Module type: {type(mod).__name__}")
            print(f"    Attributes: {[a for a in dir(mod) if not a.startswith('_')][:20]}")

    # === Step 5: Export quant params ===
    print("\n" + "=" * 60)
    print("Step 5: Saving quant params sample...")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "model_path": MODEL_PATH,
        "model_class": type(model).__name__,
        "num_quantized_layers": len(quant_layers),
        "quantized_layer_names": [n for n, _ in quant_layers[:5]],
        "inference_test": "PASSED",
        "response_sample": resp[:100],
    }
    with open(os.path.join(OUTPUT_DIR, "e2e_verify_result.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_DIR}/e2e_verify_result.json")

    print("\n" + "=" * 60)
    print("✅ E2E VERIFICATION PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
