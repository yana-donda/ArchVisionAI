from train_common import run_training

if __name__ == "__main__":
    run_training(
        model_key="efficientnet_b0",
        model_label="EfficientNet-B0",
        timm_name="efficientnet_b0",
        default_batch_size=24,
        default_lr=1e-3,
        input_size=224,
    )