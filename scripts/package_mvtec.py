"""Package verified MVTec AD models and examples as one release archive."""

import argparse
import lzma
import tarfile
from pathlib import Path

from scripts.provision_bundle import (
    MVTEC_FILES,
    MVTEC_MAX_BYTES,
    MVTEC_MAX_EXPANDED_BYTES,
    NOTICES,
    file_digest,
    verify_mvtec_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def package(artifacts: Path, destination: Path, notices: Path = REPO_ROOT) -> str:
    if not destination.name.endswith(".tar.xz"):
        raise ValueError("Release destination must end in .tar.xz.")
    sidecar = destination.with_suffix("").with_suffix(".sha256")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("Release archive or checksum already exists.")
    # Put ONNX graphs next to each other. The XZ dictionary can reuse common encoder weights.
    names = sorted(MVTEC_FILES, key=lambda name: (not name.endswith("/patchcore.onnx"), name))
    paths = {name: (notices if name in NOTICES else artifacts) / name for name in names}
    verify_mvtec_files(artifacts, notices=notices)
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    block = tarfile.BLOCKSIZE
    expanded = sum(block + ((size + block - 1) // block) * block for size in sizes.values())
    if expanded + tarfile.RECORDSIZE > MVTEC_MAX_EXPANDED_BYTES:
        raise ValueError("Release files exceed the expanded size limit.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    sidecar_created = False
    try:
        with destination.open("xb") as output:
            created = True
            with lzma.LZMAFile(output, mode="w", preset=7) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for name, path in paths.items():
                        entry = tarfile.TarInfo(name)
                        entry.size = sizes[name]
                        entry.mode = 0o644
                        with path.open("rb") as source:
                            archive.addfile(entry, source)
        if destination.stat().st_size > MVTEC_MAX_BYTES:
            raise ValueError("Release archive exceeds the compressed size limit.")
        digest = file_digest(destination)
        with sidecar.open("x", encoding="ascii", newline="\n") as output:
            sidecar_created = True
            output.write(f"{digest}\n")
    except BaseException:
        if created:
            destination.unlink()
        if sidecar_created:
            sidecar.unlink()
        raise
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = package(args.artifacts, args.output)
    print(f"{args.output.name} SHA-256: {digest}")


if __name__ == "__main__":
    main()
