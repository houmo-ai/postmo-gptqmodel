"""Exercise fixture I/O and failure paths without importing model libraries."""
import hashlib
import io
import json
import tarfile

import pytest

import _ling_fixture as fixture


@pytest.fixture
def code_archive(tmp_path, monkeypatch):
    monkeypatch.delenv("LING3_FLASH_MODEL_DIR", raising=False)
    monkeypatch.setenv("LING3_FLASH_CODE_CACHE", str(tmp_path / "cache"))
    # These bytes only test archive transport; actual model loading is covered
    # by test_ling_loader using the pinned, unmodified checkpoint source.
    files = {name: name.encode() for name in fixture.REQUIRED_FILES}
    archive = tmp_path / "code.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in files.items():
            entry = tarfile.TarInfo(f"code/{name}")
            entry.size = len(content)
            bundle.addfile(entry, io.BytesIO(content))
    manifest = {
        "url": archive.as_uri(),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fixture, "MANIFEST_PATH", path)
    return archive, manifest, files


def test_default_download_reuses_verified_cache_offline(code_archive):
    archive, _, files = code_archive
    first = fixture.ling_code_dir()
    for name, data in files.items():
        assert (first / name).read_bytes() == data
    (first / "config.json").write_text("modified by a previous test")
    archive.unlink()
    second = fixture.ling_code_dir()
    assert second != first
    assert (second / "config.json").read_bytes() == files["config.json"]


def test_explicit_directory_works_without_download(code_archive, tmp_path, monkeypatch):
    archive, _, files = code_archive
    archive.unlink()
    source = tmp_path / "local-source"
    source.mkdir()
    for name, data in files.items():
        (source / name).write_bytes(data)
    monkeypatch.setenv("LING3_FLASH_MODEL_DIR", str(source))
    assert fixture.ling_code_dir() == source.resolve()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("missing", fixture.REQUIRED_FILES)
def test_invalid_explicit_directory_fails_without_fallback(code_archive, tmp_path, monkeypatch, missing):
    _, _, files = code_archive
    for name, data in files.items():
        if name != missing:
            (tmp_path / name).write_bytes(data)
    monkeypatch.setenv("LING3_FLASH_MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=missing):
        fixture.ling_code_dir()
    assert not (tmp_path / "cache").exists()


def test_bad_archive_checksum_fails(code_archive, tmp_path):
    archive, _, _ = code_archive
    archive.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fixture.ling_code_dir()
    assert not list((tmp_path / "cache").glob("*.tar.gz"))


def test_bad_source_checksum_fails(code_archive):
    _, manifest, _ = code_archive
    manifest["files"]["config.json"]["sha256"] = "0" * 64
    fixture.MANIFEST_PATH.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256 mismatch: config.json"):
        fixture.ling_code_dir()
