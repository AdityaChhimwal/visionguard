"""HTTP inference service for a trained VisionGuard artifact."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from visionguard.artifacts import ArtifactError
from visionguard.features import FeatureExtractorError
from visionguard.images import ImageInputError
from visionguard.inference import Prediction, VisionGuardPredictor

LOGGER = logging.getLogger("visionguard.api")
DEFAULT_ARTIFACT_DIR = Path(os.environ.get("VISIONGUARD_ARTIFACT_DIR", "artifacts/bottle_v1"))
MAX_UPLOAD_BYTES = int(os.environ.get("VISIONGUARD_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


class PredictionResponse(BaseModel):
    prediction: str
    anomaly_score: float
    threshold: float
    model_version: str
    category: str


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None
    model_version: str | None = None


class MetadataResponse(BaseModel):
    model_version: str
    category: str
    feature_extractor: str
    feature_layer: str
    image_size: list[int]
    scoring_method: str
    threshold: float
    threshold_policy: dict


PredictorLoader = Callable[[Path], VisionGuardPredictor]


def create_app(
    artifact_dir: str | Path | None = None, predictor_loader: PredictorLoader = VisionGuardPredictor.from_artifact
) -> FastAPI:
    """Create a testable application that loads its model once on startup."""
    resolved_artifact_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_ARTIFACT_DIR

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        app.state.model_error = None
        try:
            LOGGER.info("Loading VisionGuard artifact from %s", resolved_artifact_dir)
            app.state.predictor = predictor_loader(resolved_artifact_dir)
            LOGGER.info("Loaded model version %s", app.state.predictor.metadata.get("model_version"))
        except (ArtifactError, FeatureExtractorError, OSError, ValueError) as exc:
            app.state.model_error = str(exc)
            LOGGER.error("Model unavailable: %s", exc)
        yield

    app = FastAPI(
        title="VisionGuard API",
        version="0.1.0",
        description="Normal-only industrial visual anomaly detection baseline.",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> VisionGuardPredictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is unavailable. Train a model and configure VISIONGUARD_ARTIFACT_DIR.",
            )
        return predictor

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        predictor = request.app.state.predictor
        if predictor is None:
            return HealthResponse(status="unavailable", detail=request.app.state.model_error)
        return HealthResponse(status="healthy", model_version=str(predictor.metadata.get("model_version")))

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(request: Request) -> MetadataResponse:
        predictor = get_predictor(request)
        metadata = predictor.metadata
        return MetadataResponse(
            model_version=str(metadata["model_version"]),
            category=str(metadata["category"]),
            feature_extractor=str(metadata["feature_extractor"]),
            feature_layer=str(metadata["feature_layer"]),
            image_size=[int(value) for value in metadata["image_size"]],
            scoring_method=str(metadata["scoring_method"]),
            threshold=float(metadata["threshold"]),
            threshold_policy=dict(metadata["threshold_policy"]),
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
        predictor = get_predictor(request)
        started = time.perf_counter()
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.",
                )
            prediction: Prediction = predictor.predict_bytes(content)
        except HTTPException:
            raise
        except ImageInputError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Inference failed")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed") from exc
        finally:
            await file.close()
        elapsed_ms = (time.perf_counter() - started) * 1000
        LOGGER.info("Prediction completed in %.1fms (%s bytes)", elapsed_ms, len(content))
        return PredictionResponse(**prediction.__dict__)

    return app


app = create_app()
