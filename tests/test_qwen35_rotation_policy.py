from pathlib import Path


def test_qwen35_readme_documents_rotation_mtp_policy():
    readme = Path("examples/quantization/examples/README.md").read_text(encoding="utf-8")

    assert "rotation 与 MTP/spec-decode 不兼容" in readme
    assert "quant.rotation: false" in readme
    assert "显式关闭 xh2modelzoo 的 MTP/spec-decode" in readme
