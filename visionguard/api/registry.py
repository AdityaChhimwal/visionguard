"""Verify configured models at startup and keep a bounded cache of ONNX sessions."""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from visionguard.api.schemas import Category, SampleInfo
from visionguard.api.settings import Settings, category_specs
from visionguard.ml.artifacts import Artifact
from visionguard.ml.images import decode_image

LOG = logging.getLogger(__name__)
SAMPLES = {"normal": "Normal example", "defect": "Defective example"}


@dataclass
class Registry:
    settings: Settings
    specs: dict = field(default_factory=dict)
    available: set[str] = field(default_factory=set, init=False)
    samples: dict[tuple[str, str], Path] = field(default_factory=dict, init=False)
    _models: OrderedDict[str, Artifact] = field(default_factory=OrderedDict, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def ready_categories(self) -> list[str]:
        return [slug for slug in self.specs if slug in self.available]

    def _load(self, slug: str) -> Artifact:
        version = self.specs[slug]["artifact"]
        return Artifact(self.settings.artifacts / version, slug, version, self.settings.threads)

    def verify_model(self, slug: str) -> None:
        # Warm a single session to check the real ONNX contract, then release it.
        self._load(slug)
        self.available.add(slug)

    def model(self, slug: str) -> Artifact | None:
        with self._lock:
            if slug not in self.available:
                return None
            cached = self._models.get(slug)
            if cached is not None:
                self._models.move_to_end(slug)
                return cached

            # Evict before opening a new graph so a one-slot host holds one
            # resident model. An in-flight request can still hold its own reference.
            if len(self._models) >= self.settings.model_cache_size:
                self._models.popitem(last=False)
            try:
                model = self._load(slug)
            except Exception as exc:
                self.available.discard(slug)
                LOG.error("Model %s no longer available (%s): %s", slug, type(exc).__name__, exc)
                return None
            self._models[slug] = model
            LOG.info("Model %s loaded into the inference cache", slug)
            return model

    def example(self, slug: str, kind: str) -> Path | None:
        if slug not in self.available:
            return None
        path = self.samples.get((slug, kind))
        return path if path is not None and path.is_file() else None

    def categories(self) -> list[Category]:
        return [
            Category(
                id=slug,
                name=self.specs[slug]["name"],
                description=self.specs[slug]["description"],
                kind=self.specs[slug].get("kind"),
                model_version=self.specs[slug]["artifact"],
                samples=[
                    SampleInfo(name=SAMPLES[kind], url=f"/api/examples/{slug}/{kind}")
                    for kind in SAMPLES if self.example(slug, kind) is not None
                ],
            )
            for slug in self.ready_categories()
        ]


def load_registry(settings: Settings) -> Registry:
    registry = Registry(settings)
    try:
        registry.specs = category_specs(settings.config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        LOG.error("Category configuration rejected (%s)", type(exc).__name__)
        return registry
    for slug, spec in registry.specs.items():
        version = spec["artifact"]
        try:
            registry.verify_model(slug)
        except Exception as exc:
            # Load failures are isolated per category. No request data is involved.
            LOG.error("Model %s/%s unavailable (%s): %s", slug, version, type(exc).__name__, exc)
            continue
        LOG.info("Model %s/%s verified", slug, version)
        for kind in SAMPLES:
            path = settings.samples / slug / f"{kind}.png"
            if not path.is_file():
                continue
            try:
                if path.stat().st_size > settings.upload_bytes:
                    raise ValueError("example exceeds upload limit")
                data = path.read_bytes()
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("example must be a PNG")
                decode_image(data, settings.max_pixels)
                registry.samples[slug, kind] = path
            except (ValueError, OSError) as exc:
                LOG.warning("Skipped example %s/%s: %s", slug, kind, exc)
    # Keep the initial selection warm after all models have been verified.
    for slug in registry.ready_categories():
        if registry.model(slug) is not None:
            break
    return registry
