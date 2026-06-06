from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def create_dataloaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    train_transform, val_transform = build_transforms()

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=train_transform),
        "val": datasets.ImageFolder(val_dir, transform=val_transform),
    }

    if len(image_datasets["train"].classes) != 2:
        raise ValueError(
            "Expected exactly 2 classes inside dataset/train, for example: "
            "architecture and not_architecture. "
            f"Found: {image_datasets['train'].classes}"
        )

    if image_datasets["train"].class_to_idx != image_datasets["val"].class_to_idx:
        raise ValueError(
            "Train and val class folders must be identical. "
            f"Train: {image_datasets['train'].class_to_idx}, "
            f"Val: {image_datasets['val'].class_to_idx}"
        )

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return dataloaders, image_datasets


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> int:
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    phase: str,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d}/{total_epochs} [{phase}]",
        unit="batch",
        leave=False,
    )

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += calculate_accuracy(outputs, labels)
        total_samples += batch_size

        current_loss = running_loss / total_samples
        current_acc = running_corrects / total_samples

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}",
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epochs: int,
) -> Tuple[nn.Module, Dict[str, list], float, int]:
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            phase="train",
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            phase="val",
            epoch=epoch,
            total_epochs=epochs,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model: val acc = {best_val_acc:.4f}")

    model.load_state_dict(best_model_state)

    return model, history, best_val_acc, best_epoch


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    class_to_idx: Dict[str, int],
    history: Dict[str, list],
    best_val_acc: float,
    best_epoch: int,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    checkpoint = {
        "model_name": "efficientnet_b0",
        "task": "architecture_or_not",
        "num_classes": 2,
        "input_size": 224,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_args": vars(args),
    }

    torch.save(checkpoint, output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "model_name": "efficientnet_b0",
        "task": "architecture_or_not",
        "num_classes": 2,
        "input_size": 224,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint": str(output_path),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Best checkpoint saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 binary classifier: architecture vs not_architecture."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset folder with train and val subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints") / "best_model_arch_or_not.pth",
        help="Path where the best checkpoint will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 on Windows if you have issues.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze EfficientNet feature extractor and train only the classifier head.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, image_datasets = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Classes: {image_datasets['train'].class_to_idx}")
    print(f"Train images: {len(image_datasets['train'])}")
    print(f"Validation images: {len(image_datasets['val'])}")

    model = build_model(
        num_classes=len(image_datasets["train"].classes),
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

    model, history, best_val_acc, best_epoch = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
    )

    save_checkpoint(
        model=model,
        output_path=args.output,
        class_to_idx=image_datasets["train"].class_to_idx,
        history=history,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        args=args,
    )


if __name__ == "__main__":
    main()
