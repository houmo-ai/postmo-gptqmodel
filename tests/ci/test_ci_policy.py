"""Scheduling and failure reporting contracts; no model imports or downloads."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, CI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = _module("select_tests")


@pytest.fixture
def manifest():
    return selector.load_manifest()


def test_docs_and_known_empty_run_only_policy(manifest):
    for paths in ([], ["AGENTS.md", "tests/ci/README.md"]):
        lanes = selector.select(manifest, paths)
        assert lanes["policy"]["tests"]
        assert all(not lanes[lane]["tests"] for lane in ("cpu", "quant", "modelzoo"))


@pytest.mark.parametrize("paths", [None, ["unclassified-script.sh"], ["setup.py"]])
def test_unknown_or_global_input_runs_all_registered_lanes(manifest, paths):
    assert all(lane["tests"] for lane in selector.select(manifest, paths).values())


def test_shared_format_selects_producer_and_consumer(manifest):
    lanes = selector.select(manifest, ["gptqmodel/quantization/config.py"])
    assert "tests/ci/test_lm_head_requantization.py" in lanes["cpu"]["tests"]
    assert "tests/ci/test_quantization_guard.py" in lanes["quant"]["tests"]
    assert "test_autoround_xh_dtype_contract.py" in lanes["modelzoo"]["tests"]


def test_target_test_change_selects_itself(manifest):
    lanes = selector.select(manifest, ["tests/ci/test_qwen3_next_recipe_api.py"])
    assert "tests/ci/test_qwen3_next_recipe_api.py" in lanes["cpu"]["tests"]
    assert not lanes["quant"]["tests"]


def test_removed_unmapped_test_cannot_silently_skip(manifest):
    assert selector.select(manifest, ["tests/deleted_test.py"])["cpu"]["tests"]


def test_force_all_does_not_bypass_invalid_manifest(tmp_path):
    root = tmp_path
    (root / "tests/ci").mkdir(parents=True)
    (root / "tests/ci/impact_policy.json").write_text('{"version": 1, "version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        selector.load_manifest(root)


@pytest.mark.parametrize("bad", ["../escape.py", "/tmp/escape.py", "!tests/a.py", "-option"])
def test_reject_bad_paths(bad):
    with pytest.raises(ValueError):
        selector._path(bad)


def test_missing_registered_test_is_error(tmp_path, manifest):
    (tmp_path / "tests/ci").mkdir(parents=True)
    policy = {key: manifest[key] for key in ("version", "always_tests", "global_patterns", "guarded_patterns")}
    (tmp_path / "tests/ci/impact_policy.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="missing local test"):
        selector.load_manifest(tmp_path)


@pytest.mark.parametrize("name,content", [("changes.txt", ""), ("changes.json", "[]")])
def test_explicit_empty_file_is_known_empty(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    assert selector.changed_paths(argparse.Namespace(changed_files_file=path))[0] == []


@pytest.mark.parametrize("ci", ["", "true"])
def test_without_external_diff_is_unknown_in_every_environment(monkeypatch, ci):
    monkeypatch.setenv("CI", ci)
    args = argparse.Namespace(changed_files_file=None)
    assert selector.changed_paths(args)[0] is None


@pytest.mark.parametrize("name", ["changes.txt", "changes.json"])
def test_external_rename_list_preserves_both_paths_and_unicode(tmp_path, name):
    paths = ["gptqmodel/量化 old.py", "gptqmodel/量化 new.py"]
    path = tmp_path / name
    path.write_text(json.dumps(paths) if name.endswith(".json") else "\n".join(paths))
    assert selector.changed_paths(argparse.Namespace(changed_files_file=path))[0] == paths


def test_runner_plan_records_external_commits(tmp_path):
    # Avoid recursive invocation: dry-run exercises the runner's plan/report
    # path; execute/check_results below cover actual subprocess results.
    command = [sys.executable, str(CI / "run.py"), "--lane", "policy", "--dry-run", "--all",
               "--gptqmodel-commit", "a" * 40, "--modelzoo-dir", str(tmp_path / "consumer"),
               "--modelzoo-commit", "b" * 40, "--report-dir", str(tmp_path)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    record = json.loads((tmp_path / "policy/run.json").read_text())
    assert record["status"] == "planned"
    assert record["gptqmodel"]["commit"] == "a" * 40
    assert record["gptqmodel"]["commit_source"] == "external"
    assert record["modelzoo"]["commit"] == "b" * 40
    assert record["selected_tests"]
    assert record["exit_code"] == 0


def test_failed_and_skipped_execution_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CI))
    runner = _module("run")
    assert runner.execute([sys.executable, "-c", "raise SystemExit(7)"], tmp_path, os.environ.copy(), 10) == 7
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite><testcase name="gpu"><skipped/></testcase></testsuite></testsuites>')
    with pytest.raises(ValueError, match="skipped"):
        runner.check_results(junit)
    junit.write_text('<testsuites/>')
    with pytest.raises(ValueError, match="no tests"):
        runner.check_results(junit)


def test_modelzoo_runner_preserves_assigned_gpu_and_consumer_checkout(tmp_path, monkeypatch, no_git):
    monkeypatch.syspath_prepend(str(CI))
    runner = _module("run")
    consumer = tmp_path / "consumer"
    tests_dir = consumer / runner.MODELZOO_CI
    tests_dir.mkdir(parents=True)
    (tests_dir / "pytest.ini").write_text("[pytest]\n")
    (tests_dir / "test_consumer_environment.py").write_text(
        "import os, sys\n"
        "def test_environment():\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,6'\n"
        f"    assert os.environ['PYTHONPATH'].split(os.pathsep)[0] == {str(consumer)!r}\n"
        f"    assert os.environ['GPTQMODEL_SOURCE_DIR'] == {str(runner.ROOT)!r}\n"
    )
    assert not (consumer / ".git").exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "unrelated-checkout"))
    monkeypatch.setenv("CI_MODELZOO_COMMIT", "b" * 40)
    monkeypatch.setattr(runner, "selection", lambda _args: {
        "lanes": {"modelzoo": {"tests": ["test_consumer_environment.py"]}}
    })
    report_dir = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["run.py", "--lane", "modelzoo", "--modelzoo-dir", str(consumer),
                                      "--report-dir", str(report_dir)])
    assert runner.main() == 0
    record = json.loads((report_dir / "modelzoo/run.json").read_text())
    assert record["status"] == "passed"
    assert record["modelzoo"]["commit"] == "b" * 40


def test_guard_rejects_corrupt_archive_and_reuses_verified_download(tmp_path):
    import hashlib

    guard = _module("test_quantization_guard")
    source = tmp_path / "source.bin"
    source.write_bytes(b"quantization fixture")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cache/archive"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        guard._download(source.as_uri(), destination, "0" * 64)
    assert not destination.exists()
    guard._download(source.as_uri(), destination, expected)
    assert destination.read_bytes() == source.read_bytes()
    source.unlink()
    guard._download(source.as_uri(), destination, expected)


def test_guard_rejects_archive_traversal(tmp_path):
    import io
    import tarfile

    guard = _module("test_quantization_guard")
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        guard._safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()


def test_project_ci_tests_do_not_enter_upstream_gpu_matrix(tmp_path):
    path = selector.ROOT / ".github/scripts/list_test_files.py"
    spec = importlib.util.spec_from_file_location("upstream_listing", path)
    listing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listing)
    (tmp_path / "ci/nested").mkdir(parents=True)
    (tmp_path / "ci/nested/test_future_gpu_guard.py").write_text("def test_case(): pass\n")
    (tmp_path / "test_existing.py").write_text("def test_case(): pass\n")
    torch_tests, _ = listing.getFiles(ignored_test_files="", tests_root=tmp_path)
    assert torch_tests == ["test_existing"]


@pytest.fixture
def isolated_repo(tmp_path):
    ci = tmp_path / "tests/ci"
    ci.mkdir(parents=True)
    (ci / "test_policy.py").write_text("def test_policy(): pass\n")
    policy = {"version": 1, "always_tests": ["tests/ci/test_policy.py"],
              "global_patterns": ["tests/ci/impact_policy.json", "gptqmodel/quantization/config.py"],
              "guarded_patterns": ["gptqmodel/**"]}
    (ci / "impact_policy.json").write_text(json.dumps(policy))
    return tmp_path


def _register(root, name, *, patterns=None, lane="cpu"):
    test = root / f"tests/ci/test_{name}.py"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.write_text("def test_behavior(): pass\n")
    declaration = test.with_suffix(".impact.json")
    declaration.write_text(json.dumps({"version": 1, "lane": lane,
                                      "patterns": patterns or [f"gptqmodel/recipes/{name}.py"]}))
    return test, declaration


@pytest.mark.parametrize("lane", ["cpu", "quant"])
def test_new_family_needs_no_policy_or_old_declaration_edit(isolated_repo, lane):
    root = isolated_repo
    _, old = _register(root, "old_family", lane=lane)
    previous = old.read_bytes()
    policy = (root / "tests/ci/impact_policy.json").read_bytes()
    before = selector.load_manifest(root)
    old_selection = selector.select(before, ["gptqmodel/recipes/old_family.py"])
    new, declaration = _register(root, "new_family", lane=lane)
    after = selector.load_manifest(root)
    new_paths = ["gptqmodel/recipes/new_family.py", new.relative_to(root).as_posix(), declaration.relative_to(root).as_posix()]
    assert selector.select(after, new_paths)[lane]["tests"] == ["tests/ci/test_new_family.py"]
    assert selector.select(after, ["gptqmodel/recipes/old_family.py"]) == old_selection
    assert old.read_bytes() == previous
    assert (root / "tests/ci/impact_policy.json").read_bytes() == policy


def test_shared_helper_selects_every_declared_consumer(isolated_repo):
    shared = "gptqmodel/utils/shared_calibration.py"
    _register(isolated_repo, "a", patterns=[shared])
    _register(isolated_repo, "b", patterns=[shared])
    lanes = selector.select(selector.load_manifest(isolated_repo), [shared])
    assert lanes["cpu"]["tests"] == ["tests/ci/test_a.py", "tests/ci/test_b.py"]


@pytest.mark.parametrize("force", [False, True])
def test_undeclared_source_fails_even_with_all(isolated_repo, force):
    manifest = selector.load_manifest(isolated_repo)
    with pytest.raises(ValueError, match="no producer test"):
        selector.select(manifest, ["gptqmodel/recipes/new_family.py"], force=force)


def test_new_ci_test_requires_own_declaration(isolated_repo):
    test, declaration = _register(isolated_repo, "new")
    declaration.unlink()
    with pytest.raises(ValueError, match="adjacent .impact.json"):
        selector.load_manifest(isolated_repo)
    test.unlink()
    selector.load_manifest(isolated_repo)


def test_declaration_cannot_select_another_test(isolated_repo):
    _, path = _register(isolated_repo, "new")
    data = json.loads(path.read_text())
    data["tests"] = ["tests/ci/test_policy.py"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid impact schema"):
        selector.load_manifest(isolated_repo)


def test_declaration_edit_selects_owner_even_when_pattern_no_longer_matches(isolated_repo):
    _, path = _register(isolated_repo, "existing", patterns=["gptqmodel/recipes/future.py"])
    manifest = selector.load_manifest(isolated_repo)
    lanes = selector.select(manifest, [path.relative_to(isolated_repo).as_posix()])
    assert lanes["cpu"]["tests"] == ["tests/ci/test_existing.py"]


def test_declaration_and_test_rename_selects_only_surviving_owner(isolated_repo):
    _register(isolated_repo, "unrelated")
    old_test, old_path = _register(isolated_repo, "before")
    old_names = [p.relative_to(isolated_repo).as_posix() for p in (old_test, old_path)]
    old_test.unlink()
    old_path.unlink()
    new_test, new_path = _register(isolated_repo, "after")
    changed = old_names + [p.relative_to(isolated_repo).as_posix() for p in (new_test, new_path)]
    lanes = selector.select(selector.load_manifest(isolated_repo), changed)
    assert lanes["cpu"]["tests"] == ["tests/ci/test_after.py"]


@pytest.mark.parametrize("pattern", ["**", "gptqmodel/recipes/*.py", "tests/ci/**", "examples/**"])
def test_overbroad_family_pattern_is_rejected(isolated_repo, pattern):
    _register(isolated_repo, "overbroad", patterns=[pattern])
    with pytest.raises(ValueError, match="overbroad"):
        selector.load_manifest(isolated_repo)


def test_real_recipe_ownership_does_not_run_other_families(manifest):
    lanes = selector.select(manifest, ["gptqmodel/recipes/qwen3_next.py"])
    assert lanes["cpu"]["tests"] == ["tests/ci/test_qwen3_next_recipe_api.py"]
    assert lanes["quant"]["tests"] == ["tests/ci/test_qwen3_next_autoround.py::test_real_autoround_quantize_save_load"]
    assert "test_transformer516_recipe_contract.py" not in lanes["modelzoo"]["tests"]


def test_consumer_declaration_cannot_replace_producer_coverage(isolated_repo):
    folder = isolated_repo / "tests/ci/modelzoo"
    folder.mkdir()
    (folder / "test_new.impact.json").write_text(json.dumps({
        "version": 1, "lane": "modelzoo", "patterns": ["gptqmodel/recipes/new.py"]}))
    manifest = selector.load_manifest(isolated_repo)
    with pytest.raises(ValueError, match="no producer test"):
        selector.select(manifest, ["gptqmodel/recipes/new.py"], force=True)


def test_legacy_test_cannot_keep_running_without_removed_declaration(isolated_repo):
    root = isolated_repo
    test = root / "tests/test_legacy.py"
    test.write_text("def test_behavior(): pass\n")
    with pytest.raises(ValueError, match="removed impact declaration"):
        selector.select(selector.load_manifest(root), ["tests/test_legacy.impact.json"])


def test_orphan_declaration_is_error(isolated_repo):
    test, _ = _register(isolated_repo, "orphan")
    test.unlink()
    with pytest.raises(ValueError, match="missing local test"):
        selector.load_manifest(isolated_repo)


def test_declaration_outside_ci_is_rejected(isolated_repo):
    test, declaration = _register(isolated_repo, "misplaced")
    for path in (test, declaration):
        path.rename(isolated_repo / "tests" / path.name)
    with pytest.raises(ValueError, match="declaration must live under tests/ci"):
        selector.load_manifest(isolated_repo)


def test_policy_test_outside_ci_is_rejected(isolated_repo):
    path = isolated_repo / "tests/ci/impact_policy.json"
    data = json.loads(path.read_text())
    data["always_tests"] = ["tests/test_policy.py"]
    (isolated_repo / "tests/ci/test_policy.py").rename(isolated_repo / "tests/test_policy.py")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="policy test must live under tests/ci"):
        selector.load_manifest(isolated_repo)


@pytest.fixture
def no_git(tmp_path, monkeypatch):
    """Fail even if a caller catches a prohibited Git subprocess failure."""
    folder = tmp_path / "no-git-bin"
    folder.mkdir()
    marker = tmp_path / "git-called"
    executable = folder / "git"
    executable.write_text('#!/bin/sh\nprintf called > "$CI_TEST_GIT_CALL_LOG"\nexit 99\n')
    executable.chmod(0o755)
    monkeypatch.setenv("CI_TEST_GIT_CALL_LOG", str(marker))
    monkeypatch.setenv("PATH", str(folder) + os.pathsep + os.environ.get("PATH", ""))
    yield
    assert not marker.exists(), "CI entrypoints must consume external input without invoking Git"


@pytest.mark.parametrize("mode", ["provided", "empty", "missing"])
def test_exported_source_runs_without_git(isolated_repo, monkeypatch, no_git, mode):
    root = isolated_repo
    _register(root, "example")
    for name in ("run.py", "select_tests.py"):
        shutil.copy2(CI / name, root / "tests/ci" / name)
    (root / "tests/ci/pytest.ini").write_text("[pytest]\n")
    assert not (root / ".git").exists()
    for key in ("CI_TEST_CHANGED_FILES_FILE", "CI_TEST_FORCE_ALL", "CI_GPTQMODEL_COMMIT", "CI_MODELZOO_COMMIT"):
        monkeypatch.delenv(key, raising=False)
    paths = None if mode == "missing" else ["gptqmodel/recipes/example.py"] if mode == "provided" else []
    if paths is not None:
        path = root / "changes.json"
        path.write_text(json.dumps(paths))
        monkeypatch.setenv("CI_TEST_CHANGED_FILES_FILE", str(path))
    if mode == "provided":
        monkeypatch.setenv("CI_GPTQMODEL_COMMIT", "a" * 40)
    command = [sys.executable, str(root / "tests/ci/run.py"), "--lane", "cpu", "--report-dir", str(root / "report")]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "report/cpu/run.json").read_text())
    assert record["status"] == ("not-required; policy passed" if mode == "empty" else "passed")
    assert record["selection"]["changed_paths"] == paths
    assert record["selected_tests"] == ([] if mode == "empty" else ["tests/ci/test_example.py"])
    assert record["gptqmodel"]["commit"] == ("a" * 40 if mode == "provided" else None)
    assert record["gptqmodel"]["commit_source"] == ("external" if mode == "provided" else "not-provided")


@pytest.mark.parametrize("content", [None, "{broken", '["ok.py", 1]', '{"paths": []}'])
def test_invalid_external_input_fails_even_with_all(tmp_path, content):
    path = tmp_path / "changes.json"
    if content is not None:
        path.write_text(content)
    command = [sys.executable, str(CI / "run.py"), "--lane", "policy", "--all",
               "--changed-files-file", str(path), "--report-dir", str(tmp_path / "report")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    record = json.loads((tmp_path / "report/policy/run.json").read_text())
    assert record["status"] == "failed"
    assert record["error"]


def test_git_ref_options_are_rejected():
    parser = argparse.ArgumentParser()
    selector.add_selection_args(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--base", "old", "--head", "new"])
    assert error.value.code == 2
