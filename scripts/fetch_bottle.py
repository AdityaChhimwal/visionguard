"""Fetch the pinned MVTec AD Bottle mirror and pinned ImageNet weights for local use."""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path

from scripts.dataset import sha256

DATASET = "https://huggingface.co/datasets/foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
WEIGHTS = (
    "https://huggingface.co/timm/resnet18.a1_in1k/resolve/"
    "491b427b45c94c7fb0e78b5474cc919aff584bbf/model.safetensors"
)
WEIGHTS_SHA256 = "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612"
EXPECTED = re.compile(r"bottle/(train/good|test/[a-z_]+|ground_truth/[a-z_]+)/\d+(_mask)?\.png")


def get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as stream:
                return stream.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("retry loop exhausted")


def fetch(root: Path) -> None:
    info_url = f"https://huggingface.co/api/datasets/foersben/mvtec-ad/revision/{REVISION}"
    info = json.loads(get(info_url))
    paths = sorted(
        item["rfilename"] for item in info["siblings"] if EXPECTED.fullmatch(item["rfilename"])
    )
    if info["sha"] != REVISION or len(set(paths)) != 355 or len(paths) != 355:
        raise ValueError("Unexpected Bottle mirror revision or image inventory.")

    def fetch_image(name: str) -> None:
        path = root / "mvtec" / name
        if path.exists() and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            return
        image = get(f"{DATASET}/resolve/{REVISION}/{name}")
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Expected a PNG: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch_image, paths))
    weights = root / "resnet18.a1_in1k.safetensors"
    if not weights.is_file() or sha256(weights) != WEIGHTS_SHA256:
        staging = weights.with_suffix(".download")
        staging.write_bytes(get(WEIGHTS))
        if sha256(staging) != WEIGHTS_SHA256:
            raise ValueError("Pretrained file checksum mismatch.")
        staging.replace(weights)
    print(f"Bottle: {len(paths)} PNGs at {REVISION}; pretrained weight SHA-256 verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    fetch(args.root)


if __name__ == "__main__":
    main()
