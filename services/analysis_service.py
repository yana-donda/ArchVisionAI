from __future__ import annotations

import asyncio
import base64
import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from PIL import Image
from flask import Request

from ml_engine import ArchVisionAnalyzer
from model_zoo import MODEL_CONFIGS, ModelType


class AnalysisService:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent)
        self.analyzer = ArchVisionAnalyzer(self.base_dir)
        self.current_mode = "efficientnet_b0"

    # ------------------------------
    # Modes / models
    # ------------------------------
    def get_current_mode(self) -> str:
        return self.current_mode

    def switch_mode(self, mode: str) -> Tuple[bool, str]:
        valid = {"efficientnet_b0", "resnet50", "ensemble"}
        if mode not in valid:
            return False, f"Невідомий режим: {mode}. Доступні: {', '.join(sorted(valid))}"
        self.current_mode = mode
        return True, "Режим оновлено"

    def get_model_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        def format_params(value: float) -> str:
            return f"{round(float(value), 1):g}M"

        if model_type == "ensemble":
            params = (
                MODEL_CONFIGS[ModelType.EFFICIENTNET_B0].params_millions
                + MODEL_CONFIGS[ModelType.RESNET50].params_millions
            )

            return {
                "type": "ensemble",
                "name": "Ensemble",
                "full_name": "Ensemble (EfficientNet-B0 + ResNet-50)",
                "params_millions": round(params, 1),
                "params": f"{MODEL_CONFIGS[ModelType.RESNET50].params_millions}M + {MODEL_CONFIGS[ModelType.EFFICIENTNET_B0].params_millions}M",
                "input_size": 224,
                "batch_size": 16,
                "description": "Усереднення ймовірностей EfficientNet-B0 та ResNet-50 для стабільнішого результату",
            }

        try:
            cfg = MODEL_CONFIGS[ModelType(model_type)]
            return {
                "type": cfg.type.value,
                "name": cfg.name,
                "full_name": cfg.name,
                "params_millions": round(float(cfg.params_millions), 1),
                "params": format_params(cfg.params_millions),
                "input_size": cfg.input_size,
                "batch_size": cfg.recommended_batch_size,
                "description": cfg.description,
            }
        except Exception:
            return None

    def get_available_modes(self) -> List[Dict[str, Any]]:
        return [
            self.get_model_info("efficientnet_b0"),
            self.get_model_info("resnet50"),
            self.get_model_info("ensemble"),
        ]

    # ------------------------------
    # Request parsing / analysis
    # ------------------------------
    def _decode_base64_image(self, image_data: str) -> bytes:
        if "," in image_data and image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        return base64.b64decode(image_data)

    def _make_thumbnail_data_url(self, image: Image.Image, max_size: int = 240) -> str:
        thumb = image.copy()
        thumb.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _read_image_from_request(self, request: Request) -> Tuple[Image.Image, Dict[str, Any], str]:
        """
        Returns: (PIL image, parsed_json_data, image_name)
        """
        json_data = request.get_json(silent=True) if request.is_json else None
        image_name = "uploaded_image"

        if "file" in request.files:
            file = request.files["file"]
            raw = file.read()
            image_name = file.filename or image_name
        elif "image" in request.files:
            file = request.files["image"]
            raw = file.read()
            image_name = file.filename or image_name
        elif json_data and json_data.get("image"):
            raw = self._decode_base64_image(json_data["image"])
        else:
            raise ValueError("Image data required")

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image, (json_data or {}), image_name

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # fallback, якщо loop вже існує
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    def analyze_request(
        self,
        request: Request,
        mode_override: Optional[str] = None,
        include_gemini: bool = False,
    ) -> Dict[str, Any]:
        image, json_data, image_name = self._read_image_from_request(request)
        use_tta = bool(json_data.get("use_tta", False))

        model_type = json_data.get("model_type") or mode_override or self.current_mode
        if model_type not in {"efficientnet_b0", "resnet50", "ensemble"}:
            model_type = self.current_mode

        result = self._run_async(
            self.analyzer.analyze_full(
                image=image,
                use_tta=use_tta,
                model_type=model_type,
                include_gemini=include_gemini,
            )
        )

        result["_meta"] = {
            "image_name": image_name,
            "image_thumbnail": self._make_thumbnail_data_url(image),
            "use_tta": use_tta,
            "model_type": model_type,
        }
        return result
    
    def analyze_gemini_request(self, request: Request) -> Dict[str, Any]:
        image, _, _ = self._read_image_from_request(request)

        gemini_result = self._run_async(
            self.analyzer.analyze_gemini_only(image)
        )

        return {
            "gemini_analysis": gemini_result,
        }

    # ------------------------------
    # Dataset images for background
    # ------------------------------
    def get_dataset_images(self, limit: int = 20) -> List[str]:
        dataset_dir = self.base_dir / "dataset"
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if not dataset_dir.exists():
            return []

        images = []

        for image_path in dataset_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                rel_path = image_path.relative_to(dataset_dir).as_posix()
                images.append(f"/api/dataset/image/{quote(rel_path)}")

        random.shuffle(images)
        return images[:limit]

    def resolve_dataset_image(self, filepath: str) -> Optional[Tuple[str, str]]:
        dataset_dir = (self.base_dir / "dataset").resolve()
        safe_path = filepath.replace("\\", "/").lstrip("/")
        full_path = (dataset_dir / safe_path).resolve()

        try:
            full_path.relative_to(dataset_dir)
        except ValueError:
            return None

        if not full_path.exists() or not full_path.is_file():
            return None

        return str(full_path.parent), full_path.name