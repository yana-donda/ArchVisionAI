from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps

from dotenv import load_dotenv
from google import genai

from model_zoo import ModelType, MODEL_CONFIGS, build_model, build_transform, load_checkpoint

load_dotenv()

logger = logging.getLogger(__name__)


class ArchVisionAnalyzer:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.data_dir = self.base_dir / "data"
        self.checkpoints_dir = self.base_dir / "checkpoints"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_mapping = self._load_class_mapping()
        self.idx_to_class = self._build_idx_to_class(self.class_mapping)
        self.architectural_styles = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]

        self.style_mapping = self._build_style_uk_mapping()
        self.geographical_data = self._load_geographical_data()
        self.ukrainian_geo_translations = self._load_ukrainian_geo_translations()
        
        self.models: Dict[str, torch.nn.Module] = {}
        self.transforms: Dict[str, Any] = {}
        self.model_load_errors: Dict[str, str] = {}

        self.gemini_keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GEMINI_API_KEY_4"),
            os.getenv("GEMINI_API_KEY"),
        ]
        self.gemini_keys = [k for k in self.gemini_keys if k]
        self.current_key_index = 0
        self.gemini_client = None
        self._gemini_ready = False
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        self._initialize_gemini()

        logger.info("ArchVisionAnalyzer initialized (device=%s)", self.device)

    # ------------------------------
    # Data loading
    # ------------------------------
    def _load_class_mapping(self) -> Dict[str, int]:
        path = self.data_dir / "class_mapping.json"
        if not path.exists():
            logger.warning("class_mapping.json not found: %s", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Підтримка двох форматів:
        # 1) {"Style name": 0, ...}
        # 2) {"0": "Style name", ...}
        if not data:
            return {}

        if all(isinstance(v, int) for v in data.values()):
            return {str(k): int(v) for k, v in data.items()}

        if all(str(k).isdigit() for k in data.keys()):
            inverted = {str(v): int(k) for k, v in data.items()}
            return inverted

        raise ValueError("Невідомий формат class_mapping.json")

    def _build_idx_to_class(self, class_mapping: Dict[str, int]) -> Dict[int, str]:
        if class_mapping:
            return {idx: style for style, idx in class_mapping.items()}

        # Fallback: якщо mapping відсутній
        fallback_styles = [
            "Achaemenid architecture",
            "American craftsman style",
            "American Foursquare architecture",
            "Ancient Egyptian architecture",
            "Art Deco architecture",
            "Art Nouveau architecture",
            "Baroque architecture",
            "Bauhaus architecture",
            "Beaux-Arts architecture",
            "Byzantine architecture",
            "Chicago school architecture",
            "Colonial architecture",
            "Deconstructivism",
            "Edwardian architecture",
            "Georgian architecture",
            "Gothic architecture",
            "Greek Revival architecture",
            "International style",
            "Novelty architecture",
            "Palladian architecture",
            "Postmodern architecture",
            "Queen Anne architecture",
            "Romanesque architecture",
            "Russian Revival architecture",
            "Tudor Revival architecture",
        ]
        return {i: style for i, style in enumerate(fallback_styles)}

    def _build_style_uk_mapping(self) -> Dict[str, str]:
        return {
            "Achaemenid architecture": "Ахеменідська архітектура",
            "American craftsman style": "Американський ремісничий стиль",
            "American Foursquare architecture": "Американська чотирикутна архітектура",
            "Ancient Egyptian architecture": "Давньоєгипетська архітектура",
            "Art Deco architecture": "Ар-деко",
            "Art Nouveau architecture": "Архітектура модерн",
            "Baroque architecture": "Бароко",
            "Bauhaus architecture": "Баухаус",
            "Beaux-Arts architecture": "Боз-Ар",
            "Byzantine architecture": "Візантійська архітектура",
            "Chicago school architecture": "Чиказька школа архітектури",
            "Colonial architecture": "Колоніальна архітектура",
            "Deconstructivism": "Деконструктивізм",
            "Edwardian architecture": "Едвардіанська архітектура",
            "Georgian architecture": "Георгіанська архітектура",
            "Gothic architecture": "Готика",
            "Greek Revival architecture": "Грецьке відродження",
            "International style": "Інтернаціональний стиль",
            "Novelty architecture": "Новаторська архітектура",
            "Palladian architecture": "Палладіанська архітектура",
            "Postmodern architecture": "Постмодернізм",
            "Queen Anne architecture": "Архітектура королеви Анни",
            "Romanesque architecture": "Романський стиль",
            "Russian Revival architecture": "Російське відродження",
            "Tudor Revival architecture": "Тюдорівське відродження",
        }

    def _load_geographical_data(self) -> Dict[str, Any]:
        path = self.data_dir / "architectural_styles_geography.json"
        if not path.exists():
            logger.warning("Geography JSON not found: %s", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Підтримка обох форматів (з обгорткою і без)
        if "architectural_styles" in data:
            return data["architectural_styles"]
        return data

    def _load_ukrainian_geo_translations(self) -> Dict[str, Dict[str, str]]:
        path = self.data_dir / "architectural_geography_ukrainian.json"
        if not path.exists():
            logger.warning("Ukrainian geo translations JSON not found: %s", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # Model loading
    def _checkpoint_candidates(self, model_type: str) -> List[Path]:
        candidates = [self.checkpoints_dir / f"{model_type}_best.pth"]

        # legacy fast_model.pth для efficientnet
        if model_type == ModelType.EFFICIENTNET_B0.value:
            candidates.append(self.data_dir / "fast_model.pth")

        return candidates

    def ensure_model_loaded(self, model_type: str) -> bool:
        if model_type in self.models:
            return True

        try:
            model_enum = ModelType(model_type)
        except ValueError:
            self.model_load_errors[model_type] = f"Непідтримуваний тип моделі: {model_type}"
            return False

        cfg = MODEL_CONFIGS[model_enum]
        model = build_model(model_enum, num_classes=len(self.idx_to_class))
        transform = build_transform(cfg.input_size)

        model_loaded = False
        last_error = None

        for ckpt in self._checkpoint_candidates(model_type):
            if not ckpt.exists():
                continue

            try:
                info = load_checkpoint(model, ckpt, device=self.device)
                logger.info("Loaded checkpoint for %s: %s", model_type, info)
                model_loaded = True
                break
            except Exception as e:
                last_error = str(e)
                logger.warning("Failed to load checkpoint %s for %s: %s", ckpt, model_type, e)

        # Навіть якщо ваги не завантажилися — модель створюємо (для структури/демо)
        model.to(self.device)
        model.eval()

        self.models[model_type] = model
        self.transforms[model_type] = transform

        if not model_loaded:
            msg = last_error or "Checkpoint не знайдено (модель створена без натренованих ваг)"
            self.model_load_errors[model_type] = msg
            logger.warning("%s: %s", model_type, msg)
        else:
            self.model_load_errors.pop(model_type, None)

        return True

    # Inference helpers
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _predict_probs(self, image: Image.Image, model_type: str) -> np.ndarray:
        self.ensure_model_loaded(model_type)

        model = self.models[model_type]
        transform = self.transforms[model_type]
        image = self._prepare_image(image)

        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        return probs

    def _generate_tta_images(self, image: Image.Image) -> List[Image.Image]:
        image = self._prepare_image(image)

        variants = [
            image,
            ImageOps.mirror(image),
            ImageEnhance.Brightness(image).enhance(1.1),
            ImageEnhance.Contrast(image).enhance(1.1),
            image.rotate(3, resample=Image.BICUBIC),
        ]
        return variants

    def _predict_probs_tta(self, image: Image.Image, model_type: str) -> np.ndarray:
        probs_all = [self._predict_probs(img, model_type) for img in self._generate_tta_images(image)]
        return np.mean(np.stack(probs_all, axis=0), axis=0)

    def _format_style_result(self, probs: np.ndarray, model_label: str, tta_used: bool = False) -> Dict[str, Any]:
        top_indices = probs.argsort()[::-1][:5]

        predictions = []
        for idx in top_indices:
            style_eng = self.idx_to_class.get(int(idx), f"class_{idx}")
            style_uk = self.style_mapping.get(style_eng, style_eng)
            predictions.append(
                {
                    "style": style_eng,
                    "style_uk": style_uk,
                    "confidence": float(probs[idx]),
                }
            )

        top_prediction = predictions[0] if predictions else {
            "style": "Невідомо",
            "style_uk": "Невідомо",
            "confidence": 0.0
        }

        geo = self._get_geographical_data(top_prediction["style"])

        result = {
            "top_prediction": top_prediction,
            "all_predictions": predictions,
            "geographical_data": geo,
            "model": model_label,
            "total_styles": len(self.architectural_styles),
        }

        if tta_used:
            result["tta_augmentations"] = 5

        return result

    def _analyze_single_model(self, image: Image.Image, model_type: str, use_tta: bool = False) -> Dict[str, Any]:
        if use_tta:
            probs = self._predict_probs_tta(image, model_type)
        else:
            probs = self._predict_probs(image, model_type)

        model_name = MODEL_CONFIGS[ModelType(model_type)].name
        style_result = self._format_style_result(probs, model_name + (" + TTA" if use_tta else ""), tta_used=use_tta)

        # Якщо модель без ваг — явно підсвітимо це
        if model_type in self.model_load_errors:
            style_result["warning"] = self.model_load_errors[model_type]

        return style_result

    def _analyze_ensemble(self, image: Image.Image, use_tta: bool = False) -> Dict[str, Any]:
        probs_eff = self._predict_probs_tta(image, ModelType.EFFICIENTNET_B0.value) if use_tta else self._predict_probs(image, ModelType.EFFICIENTNET_B0.value)
        probs_res = self._predict_probs_tta(image, ModelType.RESNET50.value) if use_tta else self._predict_probs(image, ModelType.RESNET50.value)

        probs = (probs_eff + probs_res) / 2.0
        style_result = self._format_style_result(probs, "Ensemble (EfficientNet-B0 + ResNet-50)", tta_used=use_tta)

        warnings = []
        for mt in (ModelType.EFFICIENTNET_B0.value, ModelType.RESNET50.value):
            if mt in self.model_load_errors:
                warnings.append(f"{mt}: {self.model_load_errors[mt]}")
        if warnings:
            style_result["warning"] = " | ".join(warnings)

        return style_result

    # Geography translation
    def _translate_geographical_text(self, text: str) -> str:
        if not text:
            return text

        translations = getattr(self, "ukrainian_geo_translations", {}) or {}

        for section in ("regions", "buildings", "descriptions", "building_descriptions"):
            mp = translations.get(section, {})
            if text in mp:
                return mp[text]

        logger.debug("[UA-GEO] MISS: %r", text)
        return text

    def _get_geographical_data(self, style: str) -> Dict[str, Any]:
        style_data = self.geographical_data.get(style, {})
        if not style_data:
            return {"regions": [], "famous_buildings": []}

        regions_out = []
        for region in style_data.get("regions", []):
            regions_out.append(
                {
                    "name": self._translate_geographical_text(region.get("name", "")),
                    "name_en": region.get("name", ""),
                    "center": region.get("center", []),
                    "radius_km": region.get("radius_km", 0),
                    "description": self._translate_geographical_text(region.get("description", "")),
                    "description_en": region.get("description", ""),
                }
            )

        buildings_key = "famous_buildings" if "famous_buildings" in style_data else "buildings"
        buildings_out = []
        for b in style_data.get(buildings_key, []):
            coords = b.get("coordinates", [])
            buildings_out.append(
                {
                    "name": self._translate_geographical_text(b.get("name", "")),
                    "name_en": b.get("name", ""),
                    "coordinates": coords,
                    "location": coords,  # сумісність із фронтом
                    "country": self._translate_geographical_text(b.get("country", "")),
                    "description": self._translate_geographical_text(b.get("description", "")),
                    "description_en": b.get("description", ""),
                }
            )

        return {
            "regions": regions_out,
            "famous_buildings": buildings_out,
        }

    # Public API
    async def analyze_full(self, image: Image.Image, use_tta: bool = False, model_type: str = "efficientnet_b0") -> Dict[str, Any]:
        try:
            if model_type == "ensemble":
                architectural_style = self._analyze_ensemble(image, use_tta=use_tta)
            else:
                architectural_style = self._analyze_single_model(image, model_type=model_type, use_tta=use_tta)

            gemini_result = await self._analyze_with_gemini(image)
            #gemini_analysis = await self._analyze_with_gemini_placeholder(image, architectural_style)

            return {
                "gemini_analysis": gemini_result,
                "architectural_style": architectural_style,
                "supported_styles": self.architectural_styles,
                "style_mapping": self.style_mapping,
                "geographical_data": architectural_style.get("geographical_data", {}),
                #"verification": {
                    #"overall_quality": "good" if not gemini_analysis.get("error") else "partial",
                    #"consistency_score": 0.85 if not gemini_analysis.get("error") else 0.5,
                #},
            }

        except Exception as e:
            logger.exception("Full analysis error")
            return {
                "error": f"Помилка аналізу: {str(e)}",
                "architectural_style": {
                    "top_prediction": {"style": "Невідомо", "style_uk": "Невідомо", "confidence": 0.0},
                    "all_predictions": [],
                    "geographical_data": {"regions": [], "famous_buildings": []},
                },
                "verification": {"overall_quality": "poor", "consistency_score": 0.0},
            }
            
    def _get_next_gemini_key(self):
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.gemini_keys)
        return key

    def _initialize_gemini(self):
        try:
            if not self.gemini_keys:
                self._gemini_ready = False
                return
            api_key = self._get_next_gemini_key()
            self.gemini_client = genai.Client(api_key=api_key)
            self._gemini_ready = True
        except Exception:
            self._gemini_ready = False

    async def _analyze_with_gemini(self, image):
        
        if not self._gemini_ready:
            return {"error": "Gemini недоступний (нема ключа або помилка ініціалізації)"}

        prompt = (
            """Відповідай українською мовою. Будь точним та інформативним.
            ВАЖЛИВО: без вступу, без привітань, без фраз типу 'Чудово' або 'Давайте'.
            ФОРМАТ: рівно 8 рядків. Кожен рядок починай з '1. ...', '2. ...' і так далі .
            Назву кожного пункту виділяй через **жирний текст** у Markdown.
            Ти експерт з архітектури. Проаналізуй це зображення будівлі та визнач:
            1. **Архітектурний стиль:** Точно визнач архітектурний стиль (наприклад: готика, барокко, модернізм, класицизм, арт-деко, баухаус тощо)
            2. **Назва будівлі:** Якщо це відома будівля, вкажи її назву (наприклад: Собор Паризької Богоматері, Оперний театр Сіднея тощо)
            3. **Місцезнаходження:** Вкажи місто, країну та по можливості адресу
            4. **Архітектор:** Якщо відомо, вкажи ім'я архітектора
            5. **Рік будівництва:** Коли була побудована будівля
            6. **Ключові архітектурні елементи:** Опиши характерні деталі (колони, арки, орнаменти, вікна, дах тощо)"
            7. **Історична цінність:** Чому ця будівля важлива
            8. **Сучасне використання:** Як використовується будівля зараз"""
        )

        for attempt in range(max(1, len(self.gemini_keys))):
            try:
                if attempt > 0:
                    api_key = self._get_next_gemini_key()
                    self.gemini_client = genai.Client(api_key=api_key)

                def _call():
                    return self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=[image, prompt],
                    )

                response = await asyncio.to_thread(_call)

                analysis = getattr(response, "text", None)
                if not analysis:
                    # fallback: дістаємо текст із candidates/parts
                    try:
                        analysis = response.candidates[0].content.parts[0].text
                    except Exception:
                        analysis = ""

                analysis = (analysis or "").strip()
                
                logger.info(f"Gemini text length: {len(analysis)}")
                logger.info(f"Gemini preview: {analysis[:150]}")

                if not analysis:
                    # що прийшло, але тексту нема
                    return {"error": "Gemini повернув порожній текст", "raw": str(response)}

                return {
                    "analysis": analysis,
                    "description": analysis,
                    "summary": analysis,
                    "confidence": 0.85,
                    "model": "gemini-2.5-flash",
                }
            except Exception as e:
                if attempt == len(self.gemini_keys) - 1:
                    return {"error": f"Помилка Gemini: {str(e)}"}

        return {"error": "Помилка Gemini"}

    async def _analyze_with_gemini_placeholder(self, image: Image.Image, style_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Поки що без реального Gemini-виклику (щоб проект стабільно запускався без API ключів).
        Потім можна легко замінити на реальний SDK-виклик.
        """
        top = style_result.get("top_prediction", {})
        style_name = top.get("style_uk") or top.get("style") or "Невідомо"

        if not self.gemini_keys:
            return {
                "error": "Gemini API ключ не налаштований",
                "summary": f"Ймовірний стиль: {style_name}. Детальний AI-опис тимчасово вимкнений.",
                "architectural_features": [],
                "historical_context": "",
                "cultural_significance": "",
            }

        return {
            "summary": f"Визначено стиль: {style_name}. Gemini інтеграція підключена, але в цьому етапі дипломного проєкту використовується заглушка.",
            "architectural_features": [],
            "historical_context": "",
            "cultural_significance": "",
        }