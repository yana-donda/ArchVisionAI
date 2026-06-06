from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class ArchitectureFilter:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        checkpoint_name: str = "best_model_arch_or_not.pth",
        threshold: float = 0.75,
    ):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.checkpoint_path = self.base_dir / "checkpoints" / checkpoint_name
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.class_to_idx: Dict[str, int] = {
            "architecture": 0,
            "not_architecture": 1,
        }
        self.idx_to_class: Dict[int, str] = {
            0: "architecture",
            1: "not_architecture",
        }

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._load_model()

    def _build_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 2)
        return model

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            logger.warning(
                "Architecture filter checkpoint not found: %s",
                self.checkpoint_path,
            )
            return

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            self.model = self._build_model()

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])

                class_to_idx = checkpoint.get("class_to_idx")
                if isinstance(class_to_idx, dict):
                    self.class_to_idx = {
                        str(class_name): int(index)
                        for class_name, index in class_to_idx.items()
                    }
                    self.idx_to_class = {
                        index: class_name
                        for class_name, index in self.class_to_idx.items()
                    }
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "Architecture filter loaded from %s with classes: %s",
                self.checkpoint_path,
                self.class_to_idx,
            )

        except Exception as error:
            logger.exception("Failed to load architecture filter: %s", error)
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if self.model is None:
            return {
                "checked": False,
                "is_architecture": True,
                "label": "architecture",
                "confidence": 0.0,
                "architecture_probability": 0.0,
                "not_architecture_probability": 0.0,
                "reason": "Architecture filter model is not available",
            }

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

        architecture_idx = self.class_to_idx.get("architecture", 0)
        not_architecture_idx = self.class_to_idx.get("not_architecture", 1)

        architecture_probability = float(probs[architecture_idx])
        not_architecture_probability = float(probs[not_architecture_idx])

        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = self.idx_to_class.get(predicted_idx, str(predicted_idx))
        confidence = float(probs[predicted_idx])

        is_not_architecture = (
            predicted_label == "not_architecture"
            and confidence >= self.threshold
        )

        return {
            "checked": True,
            "is_architecture": not is_not_architecture,
            "label": predicted_label,
            "confidence": confidence,
            "architecture_probability": architecture_probability,
            "not_architecture_probability": not_architecture_probability,
            "threshold": self.threshold,
        }