"""Environment settings and explicit startup category allowlist."""

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

SAFE_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def positive_env(name: str, default: int, maximum: int) -> int:
    value = int(os.environ.get(name, default))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    config: Path = Path("configs/categories.toml")
    artifacts: Path = Path("artifacts")
    samples: Path = Path("artifacts/samples")
    upload_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 12_000_000
    workers: int = 1
    threads: int = 2
    upload_seconds: int = 15
    model_cache_size: int = 1
    require_all_models: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        require_all_models = os.getenv("VG_REQUIRE_ALL_MODELS", "0")
        if require_all_models not in ("0", "1"):
            raise ValueError("VG_REQUIRE_ALL_MODELS must be 0 or 1")
        return cls(
            config=Path(os.getenv("VG_CONFIG", "configs/categories.toml")),
            artifacts=Path(os.getenv("VG_ARTIFACTS", "artifacts")),
            samples=Path(os.getenv("VG_SAMPLES", "artifacts/samples")),
            upload_bytes=positive_env("VG_UPLOAD_BYTES", 8 * 1024 * 1024, 32 * 1024 * 1024),
            max_pixels=positive_env("VG_MAX_PIXELS", 12_000_000, 32_000_000),
            workers=positive_env("VG_INFERENCE_SLOTS", 1, 4),
            threads=positive_env("VG_THREADS", 2, 8),
            upload_seconds=positive_env("VG_UPLOAD_SECONDS", 15, 120),
            model_cache_size=positive_env("VG_MODEL_CACHE_SIZE", 1, 4),
            require_all_models=require_all_models == "1",
        )


def category_specs(path: Path) -> dict:
    categories = tomllib.loads(path.read_text(encoding="utf-8"))["categories"]
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Define at least one category.")
    for category, spec in categories.items():
        if not SAFE_SLUG.fullmatch(category) or not SAFE_SLUG.fullmatch(spec["artifact"]):
            raise ValueError("Invalid category or artifact slug.")
        if not all(isinstance(spec[key], str) and spec[key] for key in ("name", "description")):
            raise ValueError("Category requires name and description.")
        if spec.get("kind") not in (None, "object", "texture"):
            raise ValueError("Category kind must be object or texture.")
    return categories
