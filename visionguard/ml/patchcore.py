"""Thin preparation adapter for Anomalib 2.7.0's PatchcoreModel."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from safetensors.torch import load_file
from torch import nn

ATOL = 1e-3
RTOL = 1e-4


def build(config: dict, weights: Path | None) -> PatchcoreModel:
    model = PatchcoreModel(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=False,
        num_neighbors=config["neighbors"],
    )
    if weights is not None:
        feature_extractor = model.feature_extractor.feature_extractor
        state = load_file(weights)
        # Timm's features_only model stops at layer3. Require every retained weight.
        feature_extractor.load_state_dict(
            {name: state[name] for name in feature_extractor.state_dict()}, strict=True
        )
    return model


@torch.no_grad()
def fit(model: PatchcoreModel, batches: Iterable[np.ndarray], ratio: float) -> dict:
    if not 0 < ratio < 1:
        raise ValueError("Coreset ratio must be between zero and one.")
    model.train()
    model.feature_extractor.eval()
    patch_count = sum(model(torch.from_numpy(batch)).shape[0] for batch in batches)
    if int(patch_count * ratio) < model.num_neighbors:
        raise ValueError("Coreset would be too small.")
    model.subsample_embedding(sampling_ratio=ratio)
    model.eval()
    rows, channels = model.memory_bank.shape
    if rows != int(patch_count * ratio) or not torch.isfinite(model.memory_bank).all():
        raise ValueError("Invalid coreset")
    return {"fit_patches": patch_count, "coreset_rows": rows, "embedding_channels": channels}


class ExportOutputs(nn.Module):
    def __init__(self, model: PatchcoreModel):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.model(image)
        return result.pred_score, result.anomaly_map


def export(model: PatchcoreModel, example: np.ndarray, destination: Path) -> None:
    # Fixed batch=1, spatial dimensions and bank size make trace branches constant.
    torch.onnx.export(
        ExportOutputs(model).eval(),
        (torch.from_numpy(example),),
        str(destination),
        input_names=["image"],
        output_names=["score", "anomaly_map"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )
