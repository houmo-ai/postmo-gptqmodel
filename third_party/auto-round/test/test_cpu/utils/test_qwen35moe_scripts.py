from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts_qwen35moe"


def test_run_no_rotate_script_only_changes_mode_and_output_names():
    rotate_script = SCRIPTS_DIR / "run_mode3_rotate_full.sh"
    no_rotate_script = SCRIPTS_DIR / "run_no_rotate.sh"

    rotate_content = rotate_script.read_text(encoding="utf-8")
    no_rotate_content = no_rotate_script.read_text(encoding="utf-8")

    expected_content = (
        rotate_content
        .replace("run_mode3_rotate_full.sh", "run_no_rotate.sh")
        .replace("Mode 3: Rotate LLM (MoE) + Rotate Vision (full) + 量化 LLM (W4G64)", "No Rotate: Quantize LLM only (W4G64)")
        .replace("适用场景: 全旋转但只量化 LLM (Qwen3.5-35B-A3B MoE)", "适用场景: 不做旋转，只量化 LLM (Qwen3.5-35B-A3B MoE)")
        .replace(
            "# 操作步骤:\n"
            "#   1. Fuse LLM RMSNorm weights → MoE experts + attention (1+weight scaling)\n"
            "#   2. Fuse Vision LayerNorm weights → adjacent linears\n"
            "#   3. Hadamard rotate 整个 Vision encoder (patch_embed, blocks, merger input)\n"
            "#   4. Rotate merger output → LLM 旋转空间\n"
            "#   5. Hadamard rotate MoE LLM backbone (gate, shared_expert, per-expert, attention)\n"
            "#   6. AutoRound 量化 LLM layers (W4G64)\n"
            "#\n"
            "# Vision encoder 已旋转但保持全精度 (不量化)\n",
            "# 操作步骤:\n"
            "#   1. 直接加载原始 Qwen3.5-MoE 模型\n"
            "#   2. 不做 Vision / merger / LLM Hadamard 旋转\n"
            "#   3. AutoRound 仅量化 LLM layers (W4G64)\n",
        )
        .replace('OUTPUT_DIR="./output/qwen35moe-mode3-rotate-full"', 'OUTPUT_DIR="./output/qwen35moe-no-rotate"')
        .replace('echo "Mode 3: Rotate MoE LLM + Full Vision → Quant LLM (W4G64)"', 'echo "No Rotate: Quantize Qwen3.5-MoE LLM only (W4G64)"')
        .replace("--mode rotate-full", "--mode llm-only")
        .replace('tee "${LOG_DIR}/qwen35moe-mode3-rotate-full.log"', 'tee "${LOG_DIR}/qwen35moe-no-rotate.log"')
    )

    assert no_rotate_content == expected_content


def test_run_no_rotate_attn8_shared8_script_only_adds_mixed_bit_overrides():
    no_rotate_script = SCRIPTS_DIR / "run_no_rotate.sh"
    mixed_script = SCRIPTS_DIR / "run_no_rotate_attn8_shared8.sh"

    no_rotate_content = no_rotate_script.read_text(encoding="utf-8")
    mixed_content = mixed_script.read_text(encoding="utf-8")

    expected_content = (
        no_rotate_content
        .replace("run_no_rotate.sh", "run_no_rotate_attn8_shared8.sh")
        .replace("No Rotate: Quantize LLM only (W4G64)", "No Rotate: Quantize LLM with Attn/Shared Expert W8")
        .replace("适用场景: 不做旋转，只量化 LLM (Qwen3.5-35B-A3B MoE)", "适用场景: 不做旋转，Attention / Shared Expert 用 W8，其余 LLM 保持 W4")
        .replace(
            "# 操作步骤:\n"
            "#   1. 直接加载原始 Qwen3.5-MoE 模型\n"
            "#   2. 不做 Vision / merger / LLM Hadamard 旋转\n"
            "#   3. AutoRound 仅量化 LLM layers (W4G64)\n",
            "# 操作步骤:\n"
            "#   1. 直接加载原始 Qwen3.5-MoE 模型\n"
            "#   2. 不做 Vision / merger / LLM Hadamard 旋转\n"
            "#   3. LLM 默认 W4G64，仅 Attention / Shared Expert 覆盖为 W8G64\n",
        )
        .replace('OUTPUT_DIR="./output/qwen35moe-no-rotate"', 'OUTPUT_DIR="./output/qwen35moe-no-rotate-attn8-shared8"')
        .replace(
            'echo "No Rotate: Quantize Qwen3.5-MoE LLM only (W4G64)"',
            'echo "No Rotate: Quantize Qwen3.5-MoE LLM (Attn/Shared Expert W8, rest W4)"',
        )
        .replace(
            'MODEL_PATH="/data01/nfs_shared/Qwen3.5-35B-A3B"',
            'MODEL_PATH="${MODEL_PATH:-/data01/nfs_shared/Qwen3.5-35B-A3B}"',
        )
        .replace('LOG_DIR="./output/logs"', 'LOG_DIR="./output/logs"\nBATCH_SIZE="${BATCH_SIZE:-1}"')
        .replace('--format auto_round:gptqmodel \\', '--format auto_round:gptqmodel \\\n    --attn_bits 8 \\\n    --shared_expert_bits 8 \\')
        .replace('--batch_size 8 \\', '--batch_size "${BATCH_SIZE}" \\')
        .replace('tee "${LOG_DIR}/qwen35moe-no-rotate.log"', 'tee "${LOG_DIR}/qwen35moe-no-rotate-attn8-shared8.log"')
    )

    assert mixed_content == expected_content


def test_run_no_rotate_attn8_shared8_qwen36_script_only_swaps_model_path_and_names():
    base_script = SCRIPTS_DIR / "run_no_rotate_attn8_shared8.sh"
    qwen36_script = SCRIPTS_DIR / "run_no_rotate_attn8_shared8_qwen36.sh"

    base_content = base_script.read_text(encoding="utf-8")
    qwen36_content = qwen36_script.read_text(encoding="utf-8")

    expected_content = (
        base_content
        .replace("run_no_rotate_attn8_shared8.sh", "run_no_rotate_attn8_shared8_qwen36.sh")
        .replace("Qwen3.5-MoE", "Qwen3.6-MoE")
        .replace("/data01/nfs_shared/Qwen3.5-35B-A3B", "/data01/datasets/Qwen3.6-35B-A3B")
        .replace('OUTPUT_DIR="./output/qwen35moe-no-rotate-attn8-shared8"', 'OUTPUT_DIR="./output/qwen36moe-no-rotate-attn8-shared8"')
        .replace('tee "${LOG_DIR}/qwen35moe-no-rotate-attn8-shared8.log"', 'tee "${LOG_DIR}/qwen36moe-no-rotate-attn8-shared8.log"')
    )

    assert qwen36_content == expected_content


def test_run_no_rotate_attn8_shared8_n256_iter400_qwen36_script_sets_its_own_calibration_and_names():
    base_script = SCRIPTS_DIR / "run_no_rotate_attn8_shared8_qwen36.sh"
    n256_iter400_script = SCRIPTS_DIR / "run_no_rotate_attn8_shared8_n256_iter400_qwen36.sh"

    base_content = base_script.read_text(encoding="utf-8")
    n256_iter400_content = n256_iter400_script.read_text(encoding="utf-8")

    expected_content = (
        base_content
        .replace(
            "run_no_rotate_attn8_shared8_qwen36.sh",
            "run_no_rotate_attn8_shared8_n256_iter400_qwen36.sh",
        )
        .replace(
            'OUTPUT_DIR="./output/qwen36moe-no-rotate-attn8-shared8"',
            'OUTPUT_DIR="./output/qwen36moe-no-rotate-attn8-shared8-n256-iter400"',
        )
        .replace("--iters 200 \\", "--iters 400 \\")
        .replace("--nsamples 128 \\", "--nsamples 256 \\")
        .replace(
            'tee "${LOG_DIR}/qwen36moe-no-rotate-attn8-shared8.log"',
            'tee "${LOG_DIR}/qwen36moe-no-rotate-attn8-shared8-n256-iter400.log"',
        )
    )

    assert n256_iter400_content == expected_content
