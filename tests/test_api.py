from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.api.main import create_app
from visionguard.api.registry import Registry
from visionguard.api.settings import Settings

SPEC = {"bottle": {"name": "Bottle", "description": "Similar setup", "artifact": "test-v1"}}


class TestOnlyModel:
    version = "test-v1"
    threshold = 0.9
    heatmap_high = 1.2

    def predict(self, image):
        assert image.mode == "RGB"
        return 1.0, np.ones((16, 16), dtype=np.float32)


def png(size=(20, 10)):
    f = BytesIO()
    Image.new("RGB", size).save(f, "PNG")
    return f.getvalue()


def test_process_alive_without_artifacts(tmp_path):
    app = create_app(Settings(artifacts=tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        assert client.get("/api/categories").json()["categories"] == []
        assert client.post("/api/predict?category=bottle", content=png()).status_code == 503


def test_discovery_prediction_and_example_delivery():
    data = png()
    registry = Registry(SPEC, {"bottle": TestOnlyModel()}, {("bottle", "normal"): data})
    with TestClient(create_app(Settings(), loader=lambda _: registry)) as client:
        category = client.get("/api/categories").json()["categories"][0]
        assert client.get("/ready").json()["categories"] == ["bottle"]
        example = client.get(category["samples"][0]["url"])
        assert example.content == data and example.headers["content-type"] == "image/png"
        result = client.post("/api/predict?category=bottle", content=example.content)
        assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
        payload = result.json()
        assert payload["label"] == "ANOMALY" and payload["anomaly_score"] == 1.0
        assert payload["threshold"] == 0.9 and payload["model_version"] == "test-v1"
        assert all(payload[key].startswith("data:image/png;base64,") for key in ("original", "overlay"))


def test_upload_rejections_before_inference():
    registry = Registry(SPEC, {"bottle": TestOnlyModel()})
    settings = Settings(upload_bytes=200, max_pixels=100)
    with TestClient(create_app(settings, loader=lambda _: registry)) as client:
        url = "/api/predict?category=bottle"
        cases = [
            ("/api/predict?category=wrong", b"a", 404, "unknown_category"),
            (url, b"", 400, "empty_upload"),
            (url, b"garbage", 422, "invalid_image"),
            (url, png(), 413, "pixel_limit"),
            (url, b"x" * 201, 413, "byte_limit"),
            (url, iter([b"x" * 101, b"x" * 100]), 413, "byte_limit"),
        ]
        for target, payload, code, reason in cases:
            result = client.post(target, content=payload)
            assert result.status_code == code
            assert result.json()["code"] == reason
