"""Fetch one complete MVTec AD category and the pinned ResNet-18 weights."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts.dataset import sha256

DATASET = "https://huggingface.co/datasets/foersben/mvtec-ad"
DATASET_API = "https://huggingface.co/api/datasets/foersben/mvtec-ad"
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
WEIGHTS = (
    "https://huggingface.co/timm/resnet18.a1_in1k/resolve/"
    "491b427b45c94c7fb0e78b5474cc919aff584bbf/model.safetensors"
)
WEIGHTS_SHA256 = "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612"
WEIGHTS_NAME = "resnet18.a1_in1k.safetensors"
# Publisher counts in order: training normals, test normals, test defects.
CATEGORY_COUNTS = {
    "bottle": (209, 20, 63),
    "cable": (224, 58, 92),
    "capsule": (219, 23, 109),
    "carpet": (280, 28, 89),
    "grid": (264, 21, 57),
    "hazelnut": (391, 40, 70),
    "leather": (245, 32, 92),
    "metal_nut": (220, 22, 93),
    "pill": (267, 26, 141),
    "screw": (320, 41, 119),
    "tile": (230, 33, 84),
    "toothbrush": (60, 12, 30),
    "transistor": (213, 60, 40),
    "wood": (247, 19, 60),
    "zipper": (240, 32, 119),
}
DATASET_PNG = re.compile(
    r"[a-z_]+/(?:train/good/[0-9]+|test/[a-z_0-9]+/[0-9]+|"
    r"ground_truth/[a-z_0-9]+/[0-9]+_mask)\.png"
)


def get(url: str) -> bytes:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as stream:
                return stream.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                raise
            wait = re.search(r"\bt=(\d+)", exc.headers.get("RateLimit", ""))
            time.sleep(min(300, int(wait[1]) + 1) if exc.code == 429 and wait else 2**attempt)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise AssertionError("retry loop exhausted")


def validate_inventory(category: str, paths: list[str]) -> None:
    names = set(paths)
    if len(names) != len(paths) or any(
        not path.startswith(f"{category}/") or not DATASET_PNG.fullmatch(path) for path in names
    ):
        raise ValueError(f"Unexpected or duplicate {category} dataset file path.")
    training = sum(path.startswith(f"{category}/train/good/") for path in names)
    normal = sum(path.startswith(f"{category}/test/good/") for path in names)
    defects = {
        path for path in names
        if path.startswith(f"{category}/test/") and not path.startswith(f"{category}/test/good/")
    }
    masks = {path for path in names if path.startswith(f"{category}/ground_truth/")}
    expected_masks = {path.replace("/test/", "/ground_truth/")[:-4] + "_mask.png" for path in defects}
    if (training, normal, len(defects)) != CATEGORY_COUNTS[category] or masks != expected_masks:
        raise ValueError(f"Incomplete {category} inventory or mismatched defect masks.")


def fetch_weights(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    weights = root / WEIGHTS_NAME
    if weights.is_file() and sha256(weights) == WEIGHTS_SHA256:
        return
    data = get(WEIGHTS)
    if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Pretrained file checksum mismatch.")
    staging = weights.with_suffix(".download")
    staging.write_bytes(data)
    staging.replace(weights)


def fetch(category: str, root: Path, workers: int = 16) -> None:
    url = f"{DATASET_API}/tree/{REVISION}/{category}?recursive=true&expand=false"
    entries = [
        entry for entry in json.loads(get(url))
        if entry["type"] == "file" and entry["path"].endswith(".png")
    ]
    validate_inventory(category, [entry["path"] for entry in entries])
    # The pinned mirror stores every dataset PNG in LFS, whose object ID is its SHA-256.
    for entry in entries:
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("lfs", {}).get("oid", "")):
            raise ValueError(f"Missing SHA-256 for {entry['path']}.")

    def fetch_image(entry: dict) -> None:
        name = entry["path"]
        digest = entry["lfs"]["oid"]
        path = root / "mvtec" / name
        if path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == digest:
            return
        image = get(f"{DATASET}/resolve/{REVISION}/{name}")
        if len(image) != entry["size"] or hashlib.sha256(image).hexdigest() != digest:
            raise ValueError(f"Unexpected bytes for {name}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".download")
        staging.write_bytes(image)
        staging.replace(path)

    fetch_weights(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(fetch_image, entries), 1):
            if index % 100 == 0:
                print(f"{category}: {index}/{len(entries)} PNGs verified", flush=True)
    print(f"{category}: {len(entries)} PNGs verified at {REVISION}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORY_COUNTS, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32.")
    fetch(args.category, args.root, args.workers)


if __name__ == "__main__":
    main()
