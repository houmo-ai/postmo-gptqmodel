"""Verified archive cache shared by CI model fixtures (standard library only)."""
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "gptqmodel-ci-fixture"})
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="download-") as work:
        partial = Path(work) / "archive.part"
        with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = _sha256(partial)
        if actual != expected_sha256:
            raise ValueError(f"CI fixture archive SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        partial.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        top_levels = {Path(member.name).parts[0] for member in bundle.getmembers() if Path(member.name).parts}
        bundle.extractall(destination, filter="data")
    if len(top_levels) == 1:
        return destination / next(iter(top_levels))
    return destination


