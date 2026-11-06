from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visionguard.api.main import create_app
from visionguard.api.settings import Settings


class StubModel:
    heatmap_high = 1.2

    def __init__(self, version, threshold):
        self.version = version
        self.threshold = threshold

    def predict(self, image):
        assert image.mode == "RGB"
        return 1.0, np.ones((16, 16), dtype=np.float32)


def png(size=(20, 10)):
    f = BytesIO()
    Image.new("RGB", size).save(f, "PNG")
    return f.getvalue()


@pytest.fixture
def model_setup(tmp_path, monkeypatch):
    config = tmp_path / "categories.toml"
    config.write_text(
        """
[categories.bottle]
name = "Bottle"
description = "Similar setup"
artifact = "bottle-test-v1"
kind = "object"

[categories.carpet]
name = "Carpet"
description = "Similar pattern"
artifact = "carpet-test-v1"
kind = "texture"

[categories.cable]
name = "Cable"
description = "Similar framing"
artifact = "cable-test-v1"
kind = "object"
""",
        encoding="utf-8",
    )
    samples = tmp_path / "samples"
    (samples / "bottle").mkdir(parents=True)
    (samples / "bottle" / "normal.png").write_bytes(png())
    (samples / "carpet").mkdir()
    (samples / "carpet" / "defect.png").write_bytes(png((12, 18)))

    unavailable = {"cable"}
    loaded = []

    def artifact(directory, category, version, threads):
        assert directory.name == version and threads > 0
        if category in unavailable:
            raise ValueError("Invalid or missing artifact")
        loaded.append(category)
        return StubModel(version, 0.9 if category == "bottle" else 1.1)

    monkeypatch.setattr("visionguard.api.registry.Artifact", artifact)
    return Settings(config=config, artifacts=tmp_path, samples=samples), loaded, unavailable


def test_process_alive_without_artifacts(tmp_path):
    app = create_app(Settings(artifacts=tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 503
        assert client.get("/api/categories").json()["categories"] == []
        assert client.post("/api/predict?category=bottle", content=png()).status_code == 503


def test_incomplete_release_does_not_pass_strict_readiness(model_setup):
    settings, _, _ = model_setup
    with TestClient(create_app(replace(settings, require_all_models=True))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert [c["id"] for c in client.get("/api/categories").json()["categories"]] == [
            "bottle", "carpet"
        ]


def test_discovery_prediction_and_example_delivery(model_setup):
    settings, _, _ = model_setup
    with TestClient(create_app(settings)) as client:
        category, carpet = client.get("/api/categories").json()["categories"]
        assert category["kind"] == "object" and carpet["kind"] == "texture"
        assert client.get("/ready").json()["categories"] == ["bottle", "carpet"]
        example = client.get(category["samples"][0]["url"])
        assert example.content == png() and example.headers["content-type"] == "image/png"
        result = client.post("/api/predict?category=bottle", content=example.content)
        assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
        payload = result.json()
        assert payload["label"] == "ANOMALY" and payload["anomaly_score"] == 1.0
        assert payload["threshold"] == 0.9 and payload["model_version"] == "bottle-test-v1"
        assert all(payload[key].startswith("data:image/png;base64,") for key in ("original", "overlay"))

        example = client.get(carpet["samples"][0]["url"])
        assert example.content == png((12, 18))
        result = client.post("/api/predict?category=carpet&preview=jpeg", content=example.content)
        assert result.status_code == 200
        payload = result.json()
        assert payload["category"] == "carpet" and payload["label"] == "NORMAL"
        assert payload["threshold"] == 1.1 and payload["model_version"] == "carpet-test-v1"
        assert all(payload[key].startswith("data:image/jpeg;base64,") for key in ("original", "overlay"))


def test_verified_categories_are_available_after_eviction(model_setup):
    settings, loaded, unavailable = model_setup
    with TestClient(create_app(settings)) as client:
        # Both graphs were tested at startup, then the first graph was warmed.
        assert loaded == ["bottle", "carpet", "bottle"]
        for slug in ("bottle", "carpet", "carpet", "bottle"):
            assert client.post(f"/api/predict?category={slug}", content=png()).status_code == 200
            assert client.get("/ready").json()["categories"] == ["bottle", "carpet"]
        assert loaded == ["bottle", "carpet", "bottle", "carpet", "bottle"]

        # A failed reload removes the affected category and its samples.
        unavailable.add("carpet")
        result = client.post("/api/predict?category=carpet", content=png())
        assert result.status_code == 503
        assert result.json()["code"] == "model_unavailable"
        assert client.get("/ready").json()["categories"] == ["bottle"]
        assert client.get("/api/examples/carpet/defect").status_code == 404


def test_upload_rejections_before_inference(model_setup):
    settings, _, _ = model_setup
    with TestClient(create_app(replace(settings, upload_bytes=200, max_pixels=100))) as client:
        url = "/api/predict?category=bottle"
        cases = [
            ("/api/predict?category=wrong", b"a", 404, "unknown_category"),
            ("/api/predict?category=cable", b"a", 503, "model_unavailable"),
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
