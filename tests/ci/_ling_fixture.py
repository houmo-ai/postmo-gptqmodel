"""Resolve real Ling checkpoint code without downloading pretrained weights."""
import json
import os
import tempfile
from pathlib import Path

from _fixture_download import _download, _safe_extract, _sha256


CI = Path(__file__).resolve().parent
MANIFEST_PATH = CI / "fixtures/ling3-flash-code-v1.json"
DEFAULT_CACHE = CI.parents[1] / "work_dirs/ci-block-fixtures/source-cache"
REQUIRED_FILES = (
    "config.json",
    "configuration_bailing_moe_v3.py",
    "modeling_bailing_moe_v3.py",
)
# Keep extracted source alive for HF imports throughout the test process.
_WORK_DIRS: list[tempfile.TemporaryDirectory] = []


def _validate_source(source: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Ling checkpoint code missing {missing} in {source}")


def ling_code_dir() -> Path:
    explicit = os.environ.get("LING3_FLASH_MODEL_DIR")
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("LING3_FLASH_MODEL_DIR must be a checkpoint code directory")
        source = Path(explicit).expanduser().resolve()
        _validate_source(source)
        print(f"CI Ling checkpoint code: explicit directory {source}", flush=True)
        return source

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cache = Path(os.environ.get("LING3_FLASH_CODE_CACHE", DEFAULT_CACHE)).expanduser()
    archive = cache / f"{manifest['sha256']}.tar.gz"
    _download(manifest["url"], archive, manifest["sha256"])
    # Re-extract verified bytes so one test/worker cannot poison another's code.
    work = tempfile.TemporaryDirectory(dir=cache, prefix="ling-code-")
    try:
        source = _safe_extract(archive, Path(work.name))
        _validate_source(source)
        for name, expected in manifest["files"].items():
            if _sha256(source / name) != expected["sha256"]:
                raise ValueError(f"Ling checkpoint code SHA-256 mismatch: {name}")
    except BaseException:
        work.cleanup()
        raise
    _WORK_DIRS.append(work)
    print(
        f"CI Ling checkpoint code: {manifest['url']} SHA-256={manifest['sha256']} "
        f"directory={source}",
        flush=True,
    )
    return source
