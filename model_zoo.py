from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import timm
from torchvision import transforms


class ModelType(str, Enum):
    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET50 = "resnet50"


@dataclass
class ModelConfig:
    type: ModelType
    name: str
    timm_name: str
    input_size: int = 224
    num_classes: int = 25
    params_millions: float = 0.0
    description: str = ""
    recommended_batch_size: int = 16


MODEL_CONFIGS: Dict[ModelType, ModelConfig] = {
    ModelType.EFFICIENTNET_B0: ModelConfig(
        type=ModelType.EFFICIENTNET_B0,
        name="EfficientNet-B0",
        timm_name="efficientnet_b0",
        input_size=224,
        num_classes=25,
        params_millions=5.3,
        description="Компактна й швидка CNN для базового класифікатора стилів",
        recommended_batch_size=32,
    ),
    ModelType.RESNET50: ModelConfig(
        type=ModelType.RESNET50,
        name="ResNet-50",
        timm_name="resnet50",
        input_size=224,
        num_classes=25,
        params_millions=25.6,
        description="Класична глибока CNN, стабільна для задач класифікації",
        recommended_batch_size=24,
    ),
}


def build_model(model_type: ModelType, num_classes: Optional[int] = None) -> torch.nn.Module:
    cfg = MODEL_CONFIGS[model_type]
    return timm.create_model(
        cfg.timm_name,
        pretrained=False,
        num_classes=num_classes or cfg.num_classes,
    )


def build_transform(input_size: int = 224) -> transforms.Compose:
    resize_size = int(input_size * 256 / 224)

    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Не вдалося знайти state_dict у checkpoint")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "").replace("_orig_mod.", "")
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint": str(checkpoint_path),
    }