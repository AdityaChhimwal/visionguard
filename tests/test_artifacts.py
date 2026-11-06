import json

import pytest

from visionguard.ml.artifacts import METADATA_FILE, MODEL_FILE, SPLIT_FILE, Artifact, checksum
from visionguard.ml.images import PREPROCESSING


def test_corrupt_graph_rejected_before_runtime_deserialization(tmp_path):
    model = b"not an ONNX graph"
    split = b"{}"
    (tmp_path / MODEL_FILE).write_bytes(model)
    (tmp_path / SPLIT_FILE).write_bytes(split)
    (tmp_path / METADATA_FILE).write_text(json.dumps({
        "schema_version": 1, "category": "bottle", "version": "v1",
        "preprocessing": PREPROCESSING, "model": {"image_size": 160},
        "calibration": {"threshold": 0.5, "heatmap_high": 0.4},
        "sha256": {MODEL_FILE: checksum(model + b"x"), SPLIT_FILE: checksum(split)},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        Artifact(tmp_path, "bottle", "v1")
