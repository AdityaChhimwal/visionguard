"""Public JSON response contracts."""

from typing import Literal

from pydantic import BaseModel


class SampleInfo(BaseModel):
    name: str
    url: str


class Category(BaseModel):
    id: str
    name: str
    description: str
    model_version: str
    samples: list[SampleInfo]


class Categories(BaseModel):
    categories: list[Category]
    max_upload_bytes: int
    max_pixels: int


class Prediction(BaseModel):
    category: str
    label: Literal["NORMAL", "ANOMALY"]
    anomaly_score: float
    threshold: float
    model_version: str
    processing_ms: float
    original: str
    overlay: str
    heatmap_range: tuple[float, float]


class Problem(BaseModel):
    code: str
    message: str
