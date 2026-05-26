from train_common import run_training

if __name__ == "__main__":
    run_training(
        model_key="resnet50",
        model_label="ResNet-50",
        timm_name="resnet50",
        default_batch_size=16,
        default_lr=1e-3,
        input_size=224,
    )