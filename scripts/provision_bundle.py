"""Install a SHA-256 pinned Bottle or MVTec AD model bundle."""

import argparse
import hashlib
import json
import lzma
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit
from zipfile import ZipFile

MAX_BYTES = 32 * 1024 * 1024
MVTEC_MAX_BYTES = 384 * 1024 * 1024
MVTEC_MAX_EXPANDED_BYTES = 512 * 1024 * 1024
BLOCK_SIZE = 1024 * 1024
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
MODEL_FILES = ("patchcore.onnx", "metadata.json", "split.json")
EXAMPLES = ("normal", "defect")
NOTICES = {"NOTICE.md", "licenses/Apache-2.0.txt"}
FILES = {
    *(f"bottle-pc-v1/{name}" for name in MODEL_FILES),
    *(f"samples/bottle/{kind}.png" for kind in EXAMPLES),
    *NOTICES,
}
MVTEC_FILES = {
    *(f"{category}-pc-v1/{name}" for category in CATEGORIES for name in MODEL_FILES),
    *(f"samples/{category}/{kind}.png" for category in CATEGORIES for kind in EXAMPLES),
    *NOTICES,
}
ZIP_MAGIC = b"PK\x03\x04"
XZ_MAGIC = b"\xfd7zXZ\x00"
PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def member_limit(name: str) -> int:
    if name.endswith("/patchcore.onnx"):
        return 64 * BLOCK_SIZE
    if name.endswith(".png"):
        return 6 * BLOCK_SIZE
    return 4 * BLOCK_SIZE


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_mvtec_files(root: Path, *, notices: Path | None = None) -> None:
    for name in MVTEC_FILES:
        path = ((notices or root) if name in NOTICES else root) / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= member_limit(name):
            raise ValueError(f"Invalid release file: {name}")
    for category in CATEGORIES:
        version = f"{category}-pc-v1"
        model = root / version
        metadata = json.loads((model / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("schema_version"), metadata.get("category"), metadata.get("version")) != (
            1,
            category,
            version,
        ):
            raise ValueError(f"Unexpected model identity: {category}")
        for filename in ("patchcore.onnx", "split.json"):
            if file_digest(model / filename) != metadata["sha256"][filename]:
                raise ValueError(f"Model file checksum mismatch: {version}/{filename}")
        for kind in EXAMPLES:
            sample = root / "samples" / category / f"{kind}.png"
            with sample.open("rb") as source:
                header = source.read(24)
            if len(header) != 24 or not header.startswith(PNG_HEADER):
                raise ValueError(f"Invalid sample PNG: {category}/{kind}")
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            if not 0 < width * height <= 4_000_000:
                raise ValueError(f"Sample exceeds Render pixel limits: {category}/{kind}")


def copy_member(source: BinaryIO, stage: Path, name: str) -> None:
    path = stage / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        shutil.copyfileobj(source, out, length=BLOCK_SIZE)


def unpack_bottle(source: BinaryIO, stage: Path) -> set[str]:
    with ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) != len(FILES) or {item.filename for item in members} != FILES:
            raise ValueError("Bundle has unexpected members.")
        if sum(item.file_size for item in members) > MAX_BYTES:
            raise ValueError("Extracted bundle exceeds size limit.")
        for name in sorted(FILES):
            with archive.open(name) as member:
                copy_member(member, stage, name)
    return FILES


class BoundedReader:
    """Bound tar headers, member bytes and padding after XZ decompression."""

    def __init__(self, source: BinaryIO, limit: int):
        self.source = source
        self.remaining = limit

    def read(self, size: int) -> bytes:
        data = self.source.read(min(size, self.remaining + 1))
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError("Extracted bundle exceeds size limit.")
        return data


def unpack_mvtec(source: BinaryIO, stage: Path) -> set[str]:
    remaining = set(MVTEC_FILES)
    with lzma.LZMAFile(source) as decompressed:
        bounded = BoundedReader(decompressed, MVTEC_MAX_EXPANDED_BYTES)
        with tarfile.open(fileobj=bounded, mode="r|") as archive:
            for member in archive:
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.name not in remaining:
                    raise ValueError("Bundle has unexpected members.")
                if not 0 < member.size <= member_limit(member.name):
                    raise ValueError(f"Bundle member exceeds size limit: {member.name}")
                source_file = archive.extractfile(member)
                if source_file is None:
                    raise ValueError(f"Unreadable bundle member: {member.name}")
                with source_file:
                    copy_member(source_file, stage, member.name)
                remaining.remove(member.name)
        # Verify the entire XZ stream, including its integrity check at EOF.
        while bounded.read(BLOCK_SIZE):
            pass
    if remaining:
        raise ValueError("Bundle is missing required members.")
    verify_mvtec_files(stage)
    return MVTEC_FILES


def publish(stage: Path, target: Path, members: Iterable[str]) -> None:
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Bundle target must be a directory.")
    if not target.exists():
        stage.rename(target)
        return
    paths = [target / name for name in sorted(members)]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError("A target file already exists; choose an empty artifact directory.")
        for parent in path.relative_to(target).parents:
            directory = target / parent
            if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
                raise ValueError(f"Bundle destination is not a directory: {directory}")

    # Preserve unrelated files in existing artifact directories, including Bottle installations.
    installed = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with (stage / path.relative_to(target)).open("rb") as source, path.open("xb") as out:
                installed.append(path)
                shutil.copyfileobj(source, out, length=BLOCK_SIZE)
    except BaseException:
        for path in installed:
            path.unlink()
        raise


def install_stream(source: BinaryIO, sha256: str, target: Path) -> None:
    if not re.fullmatch("[a-fA-F0-9]{64}", sha256):
        raise ValueError("Invalid expected SHA-256.")
    magic = source.read(len(XZ_MAGIC))
    if magic == XZ_MAGIC:
        unpack, limit = unpack_mvtec, MVTEC_MAX_BYTES
    elif magic.startswith(ZIP_MAGIC):
        unpack, limit = unpack_bottle, MAX_BYTES
    else:
        raise ValueError("Unsupported bundle format. Use the release ZIP or tar.xz.")

    digest = hashlib.sha256(magic)
    size = len(magic)
    with tempfile.TemporaryFile() as bundle:
        bundle.write(magic)
        while chunk := source.read(BLOCK_SIZE):
            size += len(chunk)
            if size > limit:
                raise ValueError("Bundle exceeds size limit.")
            digest.update(chunk)
            bundle.write(chunk)
        if digest.hexdigest() != sha256.lower():
            raise ValueError("Bundle SHA-256 mismatch.")
        bundle.seek(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".visionguard-", dir=target.parent) as work:
            stage = Path(work) / "bundle"
            stage.mkdir()
            members = unpack(bundle, stage)
            publish(stage, target, members)


def install(data: bytes, sha256: str, target: Path) -> None:
    install_stream(BytesIO(data), sha256, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--url")
    origin.add_argument("--file", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--target", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.url:
        if urlsplit(args.url).scheme != "https":
            parser.error("Use an HTTPS release URL.")
        with urllib.request.urlopen(args.url, timeout=60) as response:
            if urlsplit(response.geturl()).scheme != "https":
                parser.error("Release redirected outside HTTPS.")
            install_stream(response, args.sha256, args.target)
    else:
        with args.file.open("rb") as source:
            install_stream(source, args.sha256, args.target)


if __name__ == "__main__":
    main()
