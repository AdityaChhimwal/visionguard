import hashlib
import json
import tarfile
from base64 import b64decode
from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts.package_mvtec import package
from scripts.provision_bundle import CATEGORIES, FILES, MVTEC_FILES, file_digest, install

PNG = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1sAAAAASUVORK5CYII=")


def zipped(names):
    stream = BytesIO()
    with ZipFile(stream, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"fixture")
    return stream.getvalue()


def test_installer_requires_exact_safe_file_list(tmp_path):
    malicious = zipped(FILES | {"../elsewhere"})
    with pytest.raises(ValueError, match="unexpected members"):
        install(malicious, hashlib.sha256(malicious).hexdigest(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_installer_verifies_whole_digest_and_refuses_overwrite(tmp_path):
    payload = zipped(FILES)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install(payload, "0" * 64, tmp_path)
    digest = hashlib.sha256(payload).hexdigest()
    install(payload, digest, tmp_path)
    assert (tmp_path / "bottle-pc-v1/patchcore.onnx").read_bytes() == b"fixture"
    with pytest.raises(FileExistsError):
        install(payload, digest, tmp_path)


def write_mvtec_files(root):
    for name in sorted(MVTEC_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            path.write_bytes(PNG)
        elif path.name != "metadata.json":
            path.write_bytes(b"fixture")
    for category in CATEGORIES:
        version = f"{category}-pc-v1"
        model = root / version
        metadata = {
            "schema_version": 1,
            "category": category,
            "version": version,
            "sha256": {
                name: file_digest(model / name) for name in ("patchcore.onnx", "split.json")
            },
        }
        (model / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_mvtec_package_round_trip_includes_each_model_and_example(tmp_path):
    source = tmp_path / "source"
    write_mvtec_files(source)
    archive = tmp_path / "mvtec-ad-pc-v1.tar.xz"
    digest = package(source, archive, notices=source)
    target = tmp_path / "installed"
    install(archive.read_bytes(), digest, target)
    assert (tmp_path / "mvtec-ad-pc-v1.sha256").read_text(encoding="ascii").strip() == digest
    assert {path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()} == MVTEC_FILES
    assert (target / "samples/zipper/defect.png").read_bytes() == PNG


def test_mvtec_install_rejects_wrong_internal_model_hash_before_publishing(tmp_path):
    source = tmp_path / "source"
    write_mvtec_files(source)
    (source / "zipper-pc-v1/patchcore.onnx").write_bytes(b"wrong")
    archive = tmp_path / "invalid.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        for name in sorted(MVTEC_FILES):
            bundle.add(source / name, arcname=name)
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="Model file checksum mismatch: zipper"):
        install(archive.read_bytes(), file_digest(archive), target)
    assert not target.exists()


def test_mvtec_install_rejects_links_even_at_allowed_paths(tmp_path):
    archive = tmp_path / "invalid.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        link = tarfile.TarInfo("samples/bottle/normal.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "../elsewhere"
        bundle.addfile(link)
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="unexpected members"):
        install(archive.read_bytes(), file_digest(archive), target)
    assert not target.exists()
