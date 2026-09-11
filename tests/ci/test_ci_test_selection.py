from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_ignored_test_path_is_honored(tmp_path: Path):
    listing = _load_script("ci_list_test_files", ".github/scripts/list_test_files.py")
    tests_root = tmp_path / "tests"
    nested = tests_root / "models" / "test_mixtral.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("def test_placeholder(): pass\n", encoding="utf-8")
    (tests_root / "test_kept.py").write_text("def test_placeholder(): pass\n", encoding="utf-8")

    torch_files, _ = listing.getFiles(
        ignored_test_files="models/test_mixtral.py",
        tests_root=tests_root,
    )

    assert torch_files == ["test_kept"]
