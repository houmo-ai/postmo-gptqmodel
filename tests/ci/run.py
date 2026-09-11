#!/usr/bin/env python3
"""Portable Gerrit/Jenkins/local entrypoint for the project CI lanes."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from select_tests import LANES, ROOT, add_selection_args, selection

MODELZOO_CI = Path("benchmark_test/ci_test_transformer5.16.0")


def checkout_record(root, commit):
    """Record caller-supplied provenance without inspecting Git metadata."""
    return {"root": str(root), "commit": commit or None,
            "commit_source": "external" if commit else "not-provided"}


def versions():
    result = {"python": platform.python_version(), "executable": sys.executable}
    for package in ("torch", "transformers", "safetensors", "pytest", "auto-round", "xhquanttool"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def check_results(path):
    """A selected gate with skipped tests is incomplete, even if pytest exits 0."""
    cases = list(ET.parse(path).iter("testcase"))
    if not cases:
        raise ValueError("pytest report contains no tests")
    skipped = [case.get("name") for case in cases if case.find("skipped") is not None]
    if skipped:
        raise ValueError(f"selected CI tests skipped: {skipped}")


def execute(command, cwd, env, timeout):
    # Quantization recipes launch children. Terminate the whole test process
    # group on timeout so it cannot keep using the assigned GPU after the job.
    process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--modelzoo-dir", type=Path, default=os.environ.get("XH2MODELZOO_SOURCE_DIR"))
    parser.add_argument("--gptqmodel-commit", default=os.environ.get("CI_GPTQMODEL_COMMIT"),
                        help="Caller-supplied producer commit, recorded without resolving or verifying it")
    parser.add_argument("--modelzoo-commit", default=os.environ.get("CI_MODELZOO_COMMIT"),
                        help="Caller-supplied consumer commit, recorded without resolving or verifying it")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts/ci")
    parser.add_argument("--timeout", type=int, help="Whole pytest process tree timeout in seconds")
    add_selection_args(parser)
    args = parser.parse_args()
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    report_dir = args.report_dir.resolve() / args.lane
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {"lane": args.lane, "status": "failed", "environment": versions()}
    started = time.monotonic()
    code = 2
    try:
        report["gptqmodel"] = checkout_record(ROOT, args.gptqmodel_commit)
        if args.modelzoo_dir is not None:
            report["modelzoo"] = checkout_record(args.modelzoo_dir.expanduser().resolve(), args.modelzoo_commit)
        report["selection"] = selection(args)
        tests = report["selection"]["lanes"][args.lane]["tests"]
        report["selected_tests"] = tests
        print(json.dumps(report["selection"], ensure_ascii=True, indent=2), flush=True)
        if args.dry_run:
            report["status"] = "planned"
            code = 0
            return code
        # Preflight failures must not leave a successful report from an older
        # invocation in a reused Jenkins workspace.
        junit = report_dir / "junit.xml"
        junit.unlink(missing_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "third_party/auto-round"), env.get("PYTHONPATH", "")])
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        # Isolate from unrelated globally installed pytest plugins and options.
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        cwd = ROOT
        config = ROOT / "tests/ci/pytest.ini"
        confcutdir = ROOT / "tests/ci"
        if args.lane == "modelzoo" and tests:
            if args.modelzoo_dir is None:
                raise ValueError("modelzoo lane requires --modelzoo-dir or XH2MODELZOO_SOURCE_DIR")
            cwd = args.modelzoo_dir.expanduser().resolve()
            config = cwd / MODELZOO_CI / "pytest.ini"
            confcutdir = config.parent
            env["GPTQMODEL_SOURCE_DIR"] = str(ROOT)
            env["PYTHONPATH"] = os.pathsep.join([str(cwd), env["PYTHONPATH"]])
            tests = [str(MODELZOO_CI / test) for test in tests]
        else:
            # Even an unaffected lane emits a real, fast policy result.
            tests = sorted(set(tests + report["selection"]["lanes"]["policy"]["tests"]))
        if args.lane in ("policy", "cpu"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif args.lane == "quant" and report["selected_tests"]:
            # Preflight must use the same interpreter and source tree as pytest.
            subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'quant lane requires a dedicated CUDA device'"], env=env, cwd=cwd, check=True)
            env["GPTQMODEL_RUN_QUANT_GUARD"] = "1"
            report["quant_inputs"] = {key: env.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "GPTQMODEL_QUANT_GUARD_MODEL_DIR",
                "GPTQMODEL_QUANT_GUARD_ARCHIVE_URL", "GPTQMODEL_QUANT_GUARD_SHA256",
            )}
        for test in tests:
            if not (cwd / test.split("::", 1)[0]).is_file():
                raise ValueError(f"selected test does not exist: {cwd / test}")
        command = [sys.executable, "-m", "pytest", "-c", str(config), f"--confcutdir={confcutdir}",
                   "-o", "addopts=", "-ra", "--tb=short", "--durations=10", f"--junitxml={junit}", *tests]
        report["command"] = command
        report["cwd"] = str(cwd)
        timeout = args.timeout or (5400 if args.lane == "quant" else 1800)
        code = execute(command, cwd, env, timeout)
        if code == 0:
            check_results(junit)
            report["status"] = "passed" if report["selected_tests"] else "not-required; policy passed"
        return code
    except (ValueError, OSError, subprocess.CalledProcessError, ET.ParseError) as exc:
        report["error"] = str(exc)
        print(f"CI error: {exc}", file=sys.stderr, flush=True)
        code = 2
        return code
    finally:
        report["exit_code"] = code
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (report_dir / "run.json").write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
        print(f"CI {report['status']}: {report_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
