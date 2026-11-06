"""Load only configured local, hashed model graphs."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from visionguard.ml.images import PREPROCESSING, preprocess

MODEL_FILE = "patchcore.onnx"
SPLIT_FILE = "split.json"
METADATA_FILE = "metadata.json"
SCHEMA_VERSION = 1
MODEL_MAX_BYTES = 64 * 1024 * 1024


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Graph:
    def __init__(self, graph: bytes, image_size: int, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            graph, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.size = image_size
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or (
            inputs[0].name, inputs[0].shape, inputs[0].type
        ) != ("image", [1, 3, image_size, image_size], "tensor(float)"):
            raise ValueError("Unexpected PatchCore input contract")
        if [item.name for item in self.session.get_outputs()] != ["score", "anomaly_map"]:
            raise ValueError("Unexpected PatchCore output contract")

    def run(self, tensor: np.ndarray) -> tuple[float, np.ndarray]:
        score, anomaly_map = self.session.run(None, {"image": tensor})
        if (
            score.shape != (1,)
            or anomaly_map.shape != (1, 1, self.size, self.size)
            or not np.isfinite(score).all()
            or not np.isfinite(anomaly_map).all()
        ):
            raise ValueError("Invalid PatchCore output")
        return float(score[0]), anomaly_map[0, 0]

    def predict(self, image: Image.Image) -> tuple[float, np.ndarray]:
        return self.run(preprocess(image, self.size))


class Artifact:
    def __init__(self, directory: Path, category: str, version: str, threads: int = 2):
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if (metadata["schema_version"], metadata["category"], metadata["version"]) != (
            SCHEMA_VERSION, category, version
        ):
            raise ValueError("Artifact identity does not match configuration")
        if metadata["preprocessing"] != PREPROCESSING:
            raise ValueError("Artifact preprocessing does not match runtime")
        size = metadata["model"]["image_size"]
        if type(size) is not int or not 32 <= size <= 512:
            raise ValueError("Unsupported model input size")
        calibration = metadata["calibration"]
        self.threshold = float(calibration["threshold"])
        self.heatmap_high = float(calibration["heatmap_high"])
        if not all(math.isfinite(v) and v > 0 for v in (self.threshold, self.heatmap_high)):
            raise ValueError("Invalid artifact threshold or display scale")
        self.version = version
        self.metadata = metadata

        model = directory / MODEL_FILE
        if not 0 < model.stat().st_size <= MODEL_MAX_BYTES:
            raise ValueError("Model file size outside supported range")
        graph = model.read_bytes()
        split = (directory / SPLIT_FILE).read_bytes()
        if any(
            checksum(data) != metadata["sha256"][filename]
            for filename, data in ((MODEL_FILE, graph), (SPLIT_FILE, split))
        ):
            raise ValueError("Artifact checksum mismatch")
        self.graph = Graph(graph, size, threads)
        self.graph.run(np.zeros((1, 3, size, size), dtype=np.float32))

    def predict(self, image: Image.Image) -> tuple[float, np.ndarray]:
        return self.graph.predict(image)
