"""One same-origin web/API application with one model registry per process."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from visionguard.api.registry import Registry, load_registry
from visionguard.api.schemas import Categories, Prediction, Problem
from visionguard.api.settings import Settings
from visionguard.api.uploads import UploadProblem, read_upload
from visionguard.ml.calibration import classify
from visionguard.ml.images import ImageRejected, decode_image
from visionguard.ml.visuals import jpeg_url, png_url, render_comparison

LOG = logging.getLogger("uvicorn.error")
WEB = Path(__file__).resolve().parents[1] / "web"


def problem(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        Problem(code=code, message=message).model_dump(), status_code=status,
        headers={"Cache-Control": "no-store", **({"Retry-After": "2"} if status == 429 else {})},
    )


def infer(
    data: bytes, slug: str, registry: Registry, max_pixels: int, preview: Literal["png", "jpeg"]
) -> Prediction | None:
    start = perf_counter()
    image = decode_image(data, max_pixels)
    model = registry.model(slug)
    if model is None:
        return None
    score, anomaly_map = model.predict(image)
    original, overlay = render_comparison(image, anomaly_map, model.heatmap_high)
    encode = jpeg_url if preview == "jpeg" else png_url
    original_url, overlay_url = encode(original), encode(overlay)
    return Prediction(
        category=slug, label=classify(score, model.threshold), anomaly_score=score,
        threshold=model.threshold, model_version=model.version,
        processing_ms=round((perf_counter() - start) * 1000, 2),
        original=original_url, overlay=overlay_url, heatmap_range=(0.0, model.heatmap_high),
    )


def create_app(
    settings: Settings | None = None, loader: Callable[[Settings], Registry] = load_registry
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(service: FastAPI):
        service.state.registry = await run_in_threadpool(loader, settings)
        service.state.slots = asyncio.Semaphore(settings.workers)
        yield

    app = FastAPI(title="VisionGuard", version="1.0", lifespan=lifespan, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(
            WEB / "index.html",
            headers={
                "Cache-Control": "no-cache",
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self' blob: data:; "
                    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                ),
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": Problem}})
    def ready(request: Request):
        registry: Registry = request.app.state.registry
        categories = registry.ready_categories()
        if not categories:
            return problem(503, "model_unavailable", "No inspection model is ready.")
        if settings.require_all_models and len(categories) != len(registry.specs):
            return problem(503, "model_unavailable", "Some configured inspection models are missing.")
        return {"status": "ready", "categories": categories}

    @app.get("/api/categories", response_model=Categories)
    def categories(request: Request):
        return Categories(
            categories=request.app.state.registry.categories(),
            max_upload_bytes=settings.upload_bytes, max_pixels=settings.max_pixels,
        )

    @app.get("/api/examples/{category}/{kind}", include_in_schema=False)
    def example(category: str, kind: str, request: Request):
        path = request.app.state.registry.example(category, kind)
        if path is None:
            return problem(404, "unknown_example", "This example is unavailable.")
        return FileResponse(
            path, media_type="image/png", headers={"X-Content-Type-Options": "nosniff"}
        )

    @app.post(
        "/api/predict", response_model=Prediction,
        responses={code: {"model": Problem} for code in (400, 404, 408, 413, 422, 429, 500, 503)},
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
        }}},
    )
    async def predict(
        request: Request, category: str = "", preview: Literal["png", "jpeg"] = "png"
    ):
        registry: Registry = request.app.state.registry
        if category not in registry.specs:
            if not registry.specs:
                return problem(503, "model_unavailable", "No inspection model is ready.")
            return problem(404, "unknown_category", "Choose a supported product category.")
        if category not in registry.available:
            return problem(503, "model_unavailable", "This inspection model is unavailable.")
        slots: asyncio.Semaphore = request.app.state.slots
        if slots.locked():
            return problem(429, "busy", "The model is busy. Try again in a moment.")
        async with slots:
            try:
                data = await read_upload(request, settings.upload_bytes, settings.upload_seconds)
                result = await run_in_threadpool(
                    infer, data, category, registry, settings.max_pixels, preview
                )
            except UploadProblem as exc:
                return problem(exc.status, exc.code, str(exc))
            except ImageRejected as exc:
                return problem(413 if exc.code == "pixel_limit" else 422, exc.code, str(exc))
            except ClientDisconnect:
                return problem(400, "disconnected", "Upload was interrupted.")
            except Exception as exc:
                LOG.error("Prediction failed for %s (%s)", category, type(exc).__name__)
                return problem(500, "prediction_failed", "Inspection failed. Please try again.")
        if result is None:
            return problem(503, "model_unavailable", "This inspection model is unavailable.")
        return JSONResponse(result.model_dump(), headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


app = create_app()
