"""Install an explicitly trusted, SHA-256 pinned Bottle release bundle."""

import argparse
import hashlib
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

MAX_BYTES = 32 * 1024 * 1024
FILES = {
    "bottle-pc-v1/patchcore.onnx",
    "bottle-pc-v1/metadata.json",
    "bottle-pc-v1/split.json",
    "samples/bottle/normal.png",
    "samples/bottle/defect.png",
    "NOTICE.md",
    "licenses/Apache-2.0.txt",
}


def install(data: bytes, sha256: str, target: Path) -> None:
    if len(data) > MAX_BYTES or not re.fullmatch("[a-fA-F0-9]{64}", sha256):
        raise ValueError("Bundle too large or invalid expected SHA-256.")
    if hashlib.sha256(data).hexdigest() != sha256.lower():
        raise ValueError("Bundle SHA-256 mismatch.")
    with ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) != len(FILES) or {item.filename for item in members} != FILES:
            raise ValueError("Bundle has unexpected members.")
        if sum(item.file_size for item in members) > MAX_BYTES:
            raise ValueError("Extracted bundle exceeds size limit.")
        if any((target / name).exists() for name in FILES):
            raise FileExistsError("A target file already exists; choose an empty artifact directory.")
        for name in sorted(FILES):
            data = archive.read(name)
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                out.write(data)


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
            data = response.read(MAX_BYTES + 1)
    else:
        with args.file.open("rb") as source:
            data = source.read(MAX_BYTES + 1)
    install(data, args.sha256, args.target)


if __name__ == "__main__":
    main()
