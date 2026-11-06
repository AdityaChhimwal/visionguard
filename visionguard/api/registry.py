"""Prepare and reuse verified models and optional licensed examples."""

import logging
from dataclasses import dataclass, field

from visionguard.api.schemas import Category, SampleInfo
from visionguard.api.settings import Settings, category_specs
from visionguard.ml.artifacts import Artifact
from visionguard.ml.images import decode_image

LOG = logging.getLogger(__name__)
SAMPLES = {"normal": "Normal example", "defect": "Defective example"}


@dataclass
class Registry:
    specs: dict = field(default_factory=dict)
    models: dict[str, Artifact] = field(default_factory=dict)
    samples: dict[tuple[str, str], bytes] = field(default_factory=dict)

    def categories(self) -> list[Category]:
        return [
            Category(
                id=slug,
                name=self.specs[slug]["name"],
                description=self.specs[slug]["description"],
                model_version=model.version,
                samples=[
                    SampleInfo(name=SAMPLES[kind], url=f"/api/examples/{slug}/{kind}")
                    for category, kind in self.samples if category == slug
                ],
            )
            for slug, model in self.models.items()
        ]


def load_registry(settings: Settings) -> Registry:
    registry = Registry()
    try:
        registry.specs = category_specs(settings.config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        LOG.error("Category configuration rejected (%s)", type(exc).__name__)
        return registry
    for slug, spec in registry.specs.items():
        version = spec["artifact"]
        try:
            registry.models[slug] = Artifact(
                settings.artifacts / version, slug, version, settings.threads
            )
        except Exception as exc:
            # Load failures are isolated per category. No request data is involved.
            LOG.error("Model %s/%s unavailable (%s): %s", slug, version, type(exc).__name__, exc)
            continue
        LOG.info("Model %s/%s loaded and warmed", slug, version)
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
                registry.samples[slug, kind] = data
            except (ValueError, OSError) as exc:
                LOG.warning("Skipped example %s/%s: %s", slug, kind, exc)
    return registry
