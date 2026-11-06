import hashlib
from io import BytesIO
from zipfile import ZipFile

import pytest

from scripts.provision_bundle import FILES, install


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
