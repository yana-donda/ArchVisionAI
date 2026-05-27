from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# Utils
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(log_file: Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# Transforms
def build_train_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomChoice(
                [
                    transforms.CenterCrop(input_size),
                    transforms.RandomResizedCrop(
                        input_size,
                        scale=(0.9, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                ]
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomApply(
                [
                    transforms.RandomRotation(7),
                ],
                p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.05,
                    ),
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_val_transform(input_size: int = 224) -> transforms.Compose:
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


# Data
def build_loaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    input_size: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], int]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Не знайдено train-датасет: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Не знайдено validation-датасет: {val_dir}")

    train_dataset = ImageFolder(str(train_dir), transform=build_train_transform(input_size))
    val_dataset = ImageFolder(str(val_dir), transform=build_val_transform(input_size))

    if len(train_dataset) < 2:
        raise ValueError("У train-датасеті замало зображень. Мінімум 2.")

    if len(val_dataset) < 1:
        raise ValueError("У validation-датасеті немає зображень.")

    if len(train_dataset.classes) < 2:
        raise ValueError("Для класифікації потрібно щонайменше 2 класи.")

    train_classes = set(train_dataset.class_to_idx.keys())
    val_classes = set(val_dataset.class_to_idx.keys())

    if train_classes != val_classes:
        missing_in_val = sorted(train_classes - val_classes)
        extra_in_val = sorted(val_classes - train_classes)

        raise ValueError(
            "Класи у train і validation не збігаються. "
            f"Немає у validation: {missing_in_val}. "
            f"Зайві у validation: {extra_in_val}."
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "class_to_idx у train і validation не збігається. "
            "Перевір назви папок класів."
        )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, train_dataset.class_to_idx, len(train_dataset.classes)


# Model
def build_model(timm_name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    try:
        return timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception:
        return timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=num_classes,
        )


# Train / Validation
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is None:
                raise RuntimeError("AMP увімкнено, але scaler не створено.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{100.0 * total_correct / total_count:.2f}%",
        )

    return total_loss / total_count, 100.0 * total_correct / total_count


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="Validation", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total_count:.4f}",
            acc=f"{100.0 * total_correct / total_count:.2f}%",
        )

    return total_loss / total_count, 100.0 * total_correct / total_count


# Checkpoint
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    class_mapping: Dict[str, int],
    model_key: str,
    model_label: str,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_mapping": class_mapping,
            "model_type": model_key,
            "model_name": model_label,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "selection_metric": best_metric,
        },
        path,
    )


# Main runner
def run_training(
    model_key: str,
    model_label: str,
    timm_name: str,
    default_batch_size: int,
    default_lr: float,
    input_size: int = 224,
) -> None:
    parser = argparse.ArgumentParser(description=f"Train {model_label} with validation")

    parser.add_argument(
        "--data-dir",
        default="dataset/train",
        help="Папка з train-зображеннями у форматі ImageFolder.",
    )
    parser.add_argument(
        "--val-dir",
        default="dataset/val",
        help="Папка з validation-зображеннями у форматі ImageFolder.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--workers", type=int, default=0, help="Для Windows краще 0 або 2.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Використати pretrained ImageNet ваги.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--best-metric",
        choices=["val_acc", "val_loss"],
        default="val_acc",
        help="За якою метрикою зберігати best checkpoint.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    train_dir = (base_dir / args.data_dir).resolve()
    val_dir = (base_dir / args.val_dir).resolve()

    runs_root = base_dir / "training_runs"
    checkpoints_dir = base_dir / "checkpoints"
    data_meta_dir = base_dir / "data"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{timestamp}_{model_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "train.log"
    csv_file = run_dir / "metrics.csv"
    summary_file = run_dir / "summary.json"

    logger = setup_logger(
        log_file,
        logger_name=f"trainer_{model_key}_{timestamp}",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    logger.info("=== START %s ===", model_label)
    logger.info("Device: %s", device)
    logger.info("Train dataset: %s", train_dir)
    logger.info("Validation dataset: %s", val_dir)
    logger.info("Run dir: %s", run_dir)
    logger.info(
        "Epochs=%s, Batch=%s, LR=%s, Weight decay=%s, Best metric=%s",
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.best_metric,
    )

    train_loader, val_loader, class_to_idx, num_classes = build_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device,
        input_size=input_size,
    )

    logger.info("Classes count: %s", num_classes)
    logger.info("Train images: %s", len(train_loader.dataset))
    logger.info("Validation images: %s", len(val_loader.dataset))
    logger.info("Train batches: %s", len(train_loader))
    logger.info("Validation batches: %s", len(val_loader))

    data_meta_dir.mkdir(parents=True, exist_ok=True)
    class_mapping_path = data_meta_dir / "class_mapping.json"

    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    logger.info("Saved class mapping: %s", class_mapping_path)

    model = build_model(
        timm_name=timm_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "lr",
                "epoch_time_sec",
            ]
        )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %s/%s | "
            "train_loss=%.4f train_acc=%.2f%% | "
            "val_loss=%.4f val_acc=%.2f%% | "
            "lr=%.8f | %.1fs",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            epoch_time,
        )

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.4f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.4f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                ]
            )

        save_checkpoint(
            checkpoints_dir / f"{model_key}_last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            class_mapping=class_to_idx,
            model_key=model_key,
            model_label=model_label,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            best_metric=args.best_metric,
        )

        is_best = False

        if args.best_metric == "val_acc":
            if val_acc > best_val_acc:
                is_best = True
        else:
            if val_loss < best_val_loss:
                is_best = True

        if is_best:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            best_epoch = epoch

            save_checkpoint(
                checkpoints_dir / f"{model_key}_best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                class_mapping=class_to_idx,
                model_key=model_key,
                model_label=model_label,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_metric=args.best_metric,
            )

            logger.info(
                "New BEST checkpoint: val_loss=%.4f, val_acc=%.2f%% (epoch %s)",
                val_loss,
                val_acc,
                epoch,
            )

    summary = {
        "model_key": model_key,
        "model_name": model_label,
        "timm_name": timm_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "train_dataset": str(train_dir),
        "validation_dataset": str(val_dir),
        "validation_used": True,
        "best_metric": args.best_metric,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 6),
        "log_file": str(log_file),
        "metrics_csv": str(csv_file),
        "best_checkpoint": str(checkpoints_dir / f"{model_key}_best.pth"),
        "last_checkpoint": str(checkpoints_dir / f"{model_key}_last.pth"),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== FINISHED %s ===", model_label)
    logger.info("Best epoch: %s", best_epoch)
    logger.info("Best val_acc=%.2f%%, best val_loss=%.4f", best_val_acc, best_val_loss)