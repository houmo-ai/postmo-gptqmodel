#!/usr/bin/env python3
"""Select explicit project CI targets without importing model dependencies."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
LANES = ("policy", "cpu", "quant", "modelzoo")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid repository path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith(("!", "-")):
        raise ValueError(f"invalid repository path: {value!r}")
    return path.as_posix()


def _read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _paths(values):
    if not isinstance(values, list) or not values:
        raise ValueError("expected a non-empty path list")
    for value in values:
        _path(value)
        if any(char in value for char in "\n\r\0"):
            raise ValueError(f"invalid declared path: {value!r}")
    if len(set(values)) != len(values):
        raise ValueError("duplicate declared paths")
    return values


def _test_path(value):
    value = _path(value)
    name = PurePosixPath(value).name
    if not name.endswith(".py") or not (name.startswith("test_") or name.endswith("_test.py")):
        raise ValueError(f"not a test path: {value}")
    return value


def manifest_owner(path):
    if not path.startswith("tests/") or not path.endswith(".impact.json"):
        return None
    return _test_path(path.removesuffix(".impact.json") + ".py")


def load_manifest(root=ROOT):
    policy = _read_object(root / "tests/ci/impact_policy.json")
    if set(policy) != {"version", "global_patterns", "guarded_patterns", "always_tests"} or type(policy["version"]) is not int or policy["version"] != 1:
        raise ValueError("unsupported impact policy")
    for field in ("global_patterns", "guarded_patterns", "always_tests"):
        _paths(policy[field])
    for test in policy["always_tests"]:
        if not _test_path(test).startswith("tests/ci/"):
            raise ValueError(f"policy test must live under tests/ci: {test}")
        if not (root / test).is_file():
            raise ValueError(f"missing local test: {test}")
    targets = [{"id": "policy", "lane": "policy", "patterns": [], "tests": policy["always_tests"], "manifest": None}]
    # Source ownership must have a family/file prefix. Root-wide catch-alls
    # would silently enroll every future model in old test groups.
    broad_roots = {"gptqmodel", "gptqmodel/recipes", "gptqmodel/models",
                   "gptqmodel/models/definitions", "gptqmodel/quantization",
                   "gptqmodel/nn_modules", "gptqmodel/utils",
                   "examples", "examples/quantization", "third_party", "third_party/auto-round",
                   "tests", "tests/ci"}
    for path in sorted((root / "tests").rglob("*.impact.json")):
        relative = path.relative_to(root).as_posix()
        if not relative.startswith("tests/ci/"):
            raise ValueError(f"impact declaration must live under tests/ci: {relative}")
        owner = manifest_owner(relative)
        data = _read_object(path)
        required = {"version", "lane", "patterns"}
        if not required <= set(data) or set(data) - required - {"nodes"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError(f"invalid impact schema: {relative}")
        lane = data["lane"]
        remote = path.parent == root / "tests/ci/modelzoo"
        if lane not in ("cpu", "quant", "modelzoo") or remote != (lane == "modelzoo"):
            raise ValueError(f"invalid impact lane/location: {relative}")
        if not remote and not (root / owner).is_file():
            raise ValueError(f"missing local test: {owner}")
        patterns = _paths(data["patterns"])
        for pattern in patterns:
            prefix = pattern[:min([pattern.index(char) for char in "*?[" if char in pattern] or [len(pattern)])]
            if not prefix or prefix.rstrip("/") in broad_roots:
                raise ValueError(f"overbroad impact pattern: {pattern}; declare a family or actual shared input")
        test = path.name.removesuffix(".impact.json") + ".py" if remote else owner
        tests = [test]
        if "nodes" in data:
            nodes = _paths(data["nodes"])
            if any("/" in node or node.startswith(":") for node in nodes):
                raise ValueError(f"invalid node suffix: {relative}")
            tests = [f"{test}::{node}" for node in nodes]
        targets.append({"id": relative, "manifest": relative, "owner": owner,
                        "lane": lane, "patterns": patterns, "tests": tests})
    owners = {target.get("owner") for target in targets} | set(policy["always_tests"])
    for path in (root / "tests/ci").rglob("*.py"):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            relative = path.relative_to(root).as_posix()
            if relative not in owners:
                raise ValueError(f"CI test needs an adjacent .impact.json: {relative}")
    return {**policy, "targets": targets, "repo_root": root}


def changed_paths(args):
    if args.changed_files_file is not None:
        content = args.changed_files_file.read_text(encoding="utf-8")
        paths = json.loads(content) if args.changed_files_file.suffix == ".json" else content.splitlines()
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError("changed-file JSON must be a list of paths")
        return [_path(path) for path in paths if path], "explicit path list"
    # The caller owns checkout and patchset discovery. Missing input means
    # unknown changes even in a clean checkout or an exported source archive.
    return None, "changed-path file not provided: full registered suite"


def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def select(manifest, paths, force=False):
    targets = manifest["targets"]
    reasons = []
    paths = None if paths is None else sorted({_path(path) for path in paths})
    if force or paths is None:
        reasons.append("forced full suite" if force else "unknown diff")
    owners = {target.get("owner") for target in targets} | set(manifest["always_tests"])
    retired_owners = {manifest_owner(path) for path in paths or []
                      if manifest_owner(path) and manifest_owner(path) not in owners
                      and not (manifest["repo_root"] / manifest_owner(path)).exists()}
    for path in paths or []:
        owner = manifest_owner(path)
        if owner:
            # Changing a declaration always selects its filename-derived owner.
            # Removing ownership while retaining a test fails, even with --all.
            if owner not in owners and (manifest["repo_root"] / owner).exists():
                raise ValueError(f"removed impact declaration still has a test: {path}")
            continue
        if path in retired_owners:
            continue
        if matches(path, manifest["global_patterns"]):
            reasons.append(f"global input: {path}")
            continue
        matched = [t for t in targets if matches(path, t["patterns"]) or path == t.get("owner") or path in t["tests"]]
        if path.endswith((".md", ".rst")) or path.startswith("docs/"):
            continue
        if matches(path, manifest["guarded_patterns"]) and not any(t["lane"] in ("cpu", "quant") for t in matched):
            raise ValueError(f"no producer test declares changed source: {path}; add a focused test and .impact.json")
        if not matched:
            reasons.append(f"unmapped input: {path}")
    result = {lane: {"tests": [], "reasons": []} for lane in LANES}
    for target in targets:
        matched = [path for path in paths or [] if matches(path, target["patterns"]) or path == target.get("owner") or path == target.get("manifest") or path in target["tests"]]
        why = ["always-run policy"] if target["lane"] == "policy" else reasons or matched
        if why:
            lane = result[target["lane"]]
            lane["tests"].extend(target["tests"])
            lane["reasons"].append({"target": target["id"], "inputs": why})
    for lane in result.values():
        lane["tests"] = sorted(set(lane["tests"]))
    return result


def add_selection_args(parser):
    parser.add_argument("--changed-files-file", type=Path, default=os.environ.get("CI_TEST_CHANGED_FILES_FILE") or None,
                        help="Externally prepared changed-path list; no input selects the full registered suite")
    parser.add_argument("--all", action="store_true", default=os.environ.get("CI_TEST_FORCE_ALL") == "1")


def selection(args):
    manifest = load_manifest()
    paths, source = changed_paths(args)
    return {"diff_source": source, "changed_paths": paths, "lanes": select(manifest, paths, args.all)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_selection_args(parser)
    args = parser.parse_args()
    try:
        print(json.dumps(selection(args), ensure_ascii=True, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"CI selection error: {exc}\n")


if __name__ == "__main__":
    main()
