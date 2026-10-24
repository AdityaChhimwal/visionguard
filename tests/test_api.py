from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app
from visionguard.inference import Prediction


class FakePredictor:
    metadata = {
        "model_version": "fake-v1",
        "category": "widget",
        "feature_extractor": "vgg16_block5_global_average",
        "feature_layer": "VGG16 block5_pool followed by global average pooling",
        "image_size": [224, 224],
        "scoring_method": "minimum_euclidean_distance_between_l2_normalized_embeddings",
        "threshold": 0.5,
        "threshold_policy": {"method": "test"},
    }

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        from visionguard.images import decode_image_bytes

        decode_image_bytes(image_bytes)
        return Prediction("normal", 0.25, 0.5, "fake-v1", "widget")


def fake_loader(_: Path) -> FakePredictor:
    return FakePredictor()


def test_health_and_metadata_are_available_after_model_load() -> None:
    with TestClient(create_app("unused", predictor_loader=fake_loader)) as client:
        assert client.get("/health").json() == {"status": "healthy", "detail": None, "model_version": "fake-v1"}
        assert client.get("/metadata").status_code == 200


def test_predict_accepts_multipart_image(image_bytes: bytes) -> None:
    with TestClient(create_app("unused", predictor_loader=fake_loader)) as client:
        response = client.post("/predict", files={"file": ("sample.png", image_bytes, "image/png")})

    assert response.status_code == 200
    assert response.json()["prediction"] == "normal"
    assert response.json()["anomaly_score"] == 0.25


def test_predict_rejects_invalid_image() -> None:
    with TestClient(create_app("unused", predictor_loader=fake_loader)) as client:
        response = client.post("/predict", files={"file": ("sample.png", b"bad", "image/png")})

    assert response.status_code == 422


def test_predict_rejects_oversized_upload(monkeypatch) -> None:
    import api.main as main

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    with TestClient(create_app("unused", predictor_loader=fake_loader)) as client:
        response = client.post("/predict", files={"file": ("sample.png", b"more-than-10-bytes", "image/png")})

    assert response.status_code == 413


def test_api_reports_missing_artifact_without_fake_success() -> None:
    with TestClient(create_app("definitely-missing")) as client:
        assert client.get("/health").json()["status"] == "unavailable"
        assert client.post("/predict", files={"file": ("sample.png", b"x", "image/png")}).status_code == 503
