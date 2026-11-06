"""Package the measured Bottle graph and two pinned examples for redistribution."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.dataset import sha256
from scripts.fetch_bottle import DATASET, REVISION, get
from scripts.provision_bundle import FILES
from scripts.stage_examples import EXAMPLES

SAMPLE_SHA256 = {
    "normal": "9e7bfe658efe9b6db2607a96405b4ac6880936eaf93b89c85bcd311122ceb57b",
    "defect": "11a691a06a33dd78c5fcab6a822caa48c0d72f27f660e19f5abd9b5aa392b0eb",
}
VERIFIED_GRAPH = "f1e49d1b9ae9aac29ac12e6c3f1417302541f2009aec5fda42225e30de7aa3c1"


def package(artifact: Path, destination: Path) -> str:
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    if metadata["version"] != "bottle-pc-v1" or metadata["category"] != "bottle":
        raise ValueError("Wrong model bundle for this release.")
    for name in ("patchcore.onnx", "split.json"):
        if sha256(artifact / name) != metadata["sha256"][name]:
            raise ValueError(f"Unexpected digest for {name}.")
    if metadata["sha256"]["patchcore.onnx"] != VERIFIED_GRAPH:
        raise ValueError("Use the specific evaluated Bottle graph from the verification run.")
    content = {}
    for name in sorted(FILES):
        if name.startswith("bottle-pc-v1/"):
            content[name] = (artifact / Path(name).name).read_bytes()
        elif name.startswith("samples/"):
            kind = Path(name).stem
            sample = get(f"{DATASET}/resolve/{REVISION}/bottle/{EXAMPLES[kind]}")
            if hashlib.sha256(sample).hexdigest() != SAMPLE_SHA256[kind]:
                raise ValueError(f"Unexpected sample bytes for {kind}.")
            content[name] = sample
        else:
            content[name] = Path(name).read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as bundle:
        for name, data in content.items():
            bundle.writestr(name, data)
    return sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = package(args.artifact, args.output)
    args.output.with_suffix(".sha256").write_text(digest + "\n", encoding="ascii")
    print(f"{args.output.name} SHA-256: {digest}")


if __name__ == "__main__":
    main()
