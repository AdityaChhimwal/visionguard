"""Fit and export PatchCore, then calibrate only on held-out normals."""

import argparse
import json
import platform
import sys
import tempfile
import tomllib
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.dataset import normal_split, sha256
from visionguard.ml import patchcore
from visionguard.ml.artifacts import METADATA_FILE, MODEL_FILE, SCHEMA_VERSION, SPLIT_FILE, Graph
from visionguard.ml.calibration import calibrate
from visionguard.ml.images import PREPROCESSING, decode_image, preprocess


def tensor_for(path: Path, size: int) -> np.ndarray:
    return preprocess(decode_image(path.read_bytes()), size)


def prepare(
    config: dict, dataset: Path, weights: Path, output: Path, *, command: list[str] | None = None,
) -> dict:
    started = perf_counter()
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new output or version.")
    weight_hash = sha256(weights)
    if weight_hash != config["weights_sha256"]:
        raise ValueError("Pretrained weight checksum mismatch.")
    size = config["image_size"]
    threads = config["threads"]
    torch.set_num_threads(threads)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    manifest = normal_split(dataset, config["seed"], config["calibration_fraction"])
    fitting = manifest["fit"]
    batch_size = config["batch_size"]
    batches = (
        np.concatenate([tensor_for(dataset / path, size) for path in fitting[i:i + batch_size]])
        for i in range(0, len(fitting), batch_size)
    )
    model = patchcore.build(config, weights)
    counts = patchcore.fit(model, batches, config["coreset_ratio"])
    print(f"Fit coreset: {counts}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        directory = Path(staging) / "bundle"
        directory.mkdir()
        example = tensor_for(dataset / manifest["calibration"][0], size)
        patchcore.export(model, example, directory / MODEL_FILE)
        graph = Graph((directory / MODEL_FILE).read_bytes(), size, threads)
        scores, peaks = [], []
        score_delta, map_delta = 0.0, 0.0
        pytorch_model = patchcore.ExportOutputs(model)
        with torch.inference_mode():
            for name in manifest["calibration"]:
                tensor = tensor_for(dataset / name, size)
                score, heatmap = graph.run(tensor)
                expected_score, expected_map = pytorch_model(torch.from_numpy(tensor))
                expected_map = expected_map.numpy()[0, 0]
                expected_score = float(expected_score.item())
                np.testing.assert_allclose(score, expected_score, rtol=patchcore.RTOL, atol=patchcore.ATOL)
                np.testing.assert_allclose(
                    heatmap, expected_map, rtol=patchcore.RTOL, atol=patchcore.ATOL
                )
                score_delta = max(score_delta, abs(score - expected_score))
                map_delta = max(map_delta, float(np.abs(heatmap - expected_map).max()))
                scores.append(score)
                peaks.append(float(heatmap.max()))

        (directory / SPLIT_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        meta = {
            "schema_version": SCHEMA_VERSION,
            "category": config["category"],
            "version": config["version"],
            "model": config | counts,
            "preprocessing": PREPROCESSING,
            "weights": {
                "url": f"https://huggingface.co/timm/{config['backbone']}/tree/{config['weights_revision']}",
                "sha256": weight_hash,
            },
            "dependencies": {
                name: version(name) for name in (
                    "anomalib", "torch", "torchvision", "timm", "onnx", "onnxruntime",
                    "numpy", "pillow", "safetensors", "scikit-learn"
                )
            },
            "calibration": calibrate(scores, peaks, config["false_positive_target"]),
            "export_parity": {
                "calibration_images": len(scores),
                "rtol": patchcore.RTOL, "atol": patchcore.ATOL,
                "max_abs_score_delta": score_delta, "max_abs_map_delta": map_delta,
            },
            "sha256": {name: sha256(directory / name) for name in (MODEL_FILE, SPLIT_FILE)},
            "preparation": {
                "command": command or ["python", "-m", "scripts.prepare", *sys.argv[1:]],
                "os": platform.platform(),
                "cpu": platform.processor(),
                "seconds": round(perf_counter() - started, 3),
            },
        }
        (directory / METADATA_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        directory.rename(output)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset", "weights", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare(config, args.dataset, args.weights, args.output), indent=2))


if __name__ == "__main__":
    main()
