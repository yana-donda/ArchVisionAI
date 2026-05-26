from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

from model_zoo import ModelType, MODEL_CONFIGS, build_model, build_transform, load_checkpoint


BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "dataset" / "test"
CLASS_MAPPING_PATH = BASE_DIR / "data" / "class_mapping.json"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

OUTPUT_CSV = BASE_DIR / "test_results" / "model_test_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_class_mapping():
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_test_images(class_mapping):
    samples = []

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Не знайдено тестову папку: {TEST_DIR}")

    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        if class_name not in class_mapping:
            raise ValueError(
                f"Клас '{class_name}' є в dataset/test, але його немає в class_mapping.json"
            )

        label = int(class_mapping[class_name])

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    if not samples:
        raise ValueError("У dataset/test не знайдено зображень.")

    return samples


def load_one_model(model_type, num_classes, device):
    model_enum = ModelType(model_type)

    model = build_model(model_enum, num_classes=num_classes)

    checkpoint_path = CHECKPOINTS_DIR / f"{model_type}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Не знайдено checkpoint: {checkpoint_path}")

    load_checkpoint(model, checkpoint_path, device=str(device))

    model.to(device)
    model.eval()

    transform = build_transform(MODEL_CONFIGS[model_enum].input_size)

    return model, transform


def make_tta_images(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        bicubic = Image.Resampling.BICUBIC
    except AttributeError:
        bicubic = Image.BICUBIC

    return [
        image,
        ImageOps.mirror(image),
        ImageEnhance.Brightness(image).enhance(1.1),
        ImageEnhance.Contrast(image).enhance(1.1),
        image.rotate(3, resample=bicubic),
    ]


@torch.no_grad()
def predict_probs_single_model(model, transform, image, device, use_tta=False):
    if image.mode != "RGB":
        image = image.convert("RGB")

    images = make_tta_images(image) if use_tta else [image]

    probs_list = []

    for img in images:
        tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)

        probs_list.append(probs.squeeze(0).detach().cpu().numpy())

    return np.mean(np.stack(probs_list, axis=0), axis=0)


@torch.no_grad()
def predict_probs_ensemble(
    efficientnet_model,
    efficientnet_transform,
    resnet_model,
    resnet_transform,
    image,
    device,
    use_tta=False,
):
    probs_eff = predict_probs_single_model(
        efficientnet_model,
        efficientnet_transform,
        image,
        device,
        use_tta=use_tta,
    )

    probs_res = predict_probs_single_model(
        resnet_model,
        resnet_transform,
        image,
        device,
        use_tta=use_tta,
    )

    return (probs_eff + probs_res) / 2.0


def calculate_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return float((y_true == y_pred).mean())


def calculate_f1_macro(y_true, y_pred, num_classes):
    f1_scores = []

    for class_idx in range(num_classes):
        tp = 0
        fp = 0
        fn = 0

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == class_idx and pred_label == class_idx:
                tp += 1
            elif true_label != class_idx and pred_label == class_idx:
                fp += 1
            elif true_label == class_idx and pred_label != class_idx:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_mode(mode_name, samples, num_classes, device, predict_function):
    y_true = []
    y_pred = []

    sync_cuda(device)
    start_time = time.perf_counter()

    for image_path, true_label in tqdm(samples, desc=mode_name):
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        probs = predict_function(image)
        pred_label = int(np.argmax(probs))

        y_true.append(true_label)
        y_pred.append(pred_label)

    sync_cuda(device)
    end_time = time.perf_counter()

    accuracy = calculate_accuracy(y_true, y_pred)
    f1_macro = calculate_f1_macro(y_true, y_pred, num_classes)

    avg_time_per_image = (end_time - start_time) / len(samples)

    return {
        "режим": mode_name,
        "точність": round(accuracy * 100, 4),
        "f1 macro": round(f1_macro * 100, 4),
        "час": round(avg_time_per_image, 6),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_mapping = load_class_mapping()
    num_classes = len(class_mapping)

    samples = collect_test_images(class_mapping)

    print(f"Класів: {num_classes}")
    print(f"Тестових зображень: {len(samples)}")

    efficientnet_model, efficientnet_transform = load_one_model(
        "efficientnet_b0",
        num_classes,
        device,
    )

    resnet_model, resnet_transform = load_one_model(
        "resnet50",
        num_classes,
        device,
    )

    results = []

    results.append(
        evaluate_mode(
            mode_name="EfficientNet-B0",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_single_model(
                efficientnet_model,
                efficientnet_transform,
                image,
                device,
                use_tta=False,
            ),
        )
    )

    results.append(
        evaluate_mode(
            mode_name="EfficientNet-B0 + TTA",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_single_model(
                efficientnet_model,
                efficientnet_transform,
                image,
                device,
                use_tta=True,
            ),
        )
    )

    results.append(
        evaluate_mode(
            mode_name="ResNet-50",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_single_model(
                resnet_model,
                resnet_transform,
                image,
                device,
                use_tta=False,
            ),
        )
    )

    results.append(
        evaluate_mode(
            mode_name="ResNet-50 + TTA",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_single_model(
                resnet_model,
                resnet_transform,
                image,
                device,
                use_tta=True,
            ),
        )
    )

    results.append(
        evaluate_mode(
            mode_name="Ensemble",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_ensemble(
                efficientnet_model,
                efficientnet_transform,
                resnet_model,
                resnet_transform,
                image,
                device,
                use_tta=False,
            ),
        )
    )

    results.append(
        evaluate_mode(
            mode_name="Ensemble + TTA",
            samples=samples,
            num_classes=num_classes,
            device=device,
            predict_function=lambda image: predict_probs_ensemble(
                efficientnet_model,
                efficientnet_transform,
                resnet_model,
                resnet_transform,
                image,
                device,
                use_tta=True,
            ),
        )
    )
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["режим", "точність", "f1 macro", "час"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nГотово. Результати збережено у файл:")
    print(OUTPUT_CSV)

    print("\nРезультати:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()