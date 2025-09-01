import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ---- Parse CLI ----
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning model (cats vs dogs)."
    )

    # Required paths
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed dataset root (expects train/ and val/ subdirs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/",
        help="Directory to save trained models and logs (default: models/).",
    )

    # Model / training knobs
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mobilenet_v2", "efficientnet_b0"],
        default="mobilenet_v2",
        help="Base CNN to use for transfer learning (default: mobilenet_v2).",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Image size (default: 224).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--epochs", type=int, default=12, help="Total epochs (default: 12).")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=4,
        help="Epochs to train classifier head with backbone frozen (default: 4).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for head training (default: 1e-3).")
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="LR for fine-tuning backbone (default: 1e-4).",
    )
    parser.add_argument(
        "--finetune-top",
        type=int,
        default=50,
        help="Number of top backbone layers to unfreeze (default: 50). Use 0 to skip fine-tune.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Enable mixed precision (Apple Silicon with tensorflow-metal).",
    )

    # Legacy compatibility (unused but harmless)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


# ---- Validation & setup ----
def validate_args(args):
    data_root = Path(os.path.expanduser(args.data)).resolve()
    out_root = Path(os.path.expanduser(args.output)).resolve()

    if not data_root.exists():
        sys.exit(f"[ERROR] Data path does not exist: {data_root}")
    if not (data_root / "train").exists() or not (data_root / "val").exists():
        sys.exit("[ERROR] Data path must contain 'train/' and 'val/' subdirectories.")

    if args.img_size <= 0:
        sys.exit("[ERROR] --img-size must be > 0.")
    if args.batch_size <= 0:
        sys.exit("[ERROR] --batch-size must be > 0.")
    if args.epochs <= 0:
        sys.exit("[ERROR] --epochs must be > 0.")
    if args.freeze_epochs < 0 or args.freeze_epochs > args.epochs:
        sys.exit("[ERROR] --freeze-epochs must be between 0 and --epochs.")
    if args.finetune_top < 0:
        sys.exit("[ERROR] --finetune-top must be >= 0.")

    out_root.mkdir(parents=True, exist_ok=True)
    return data_root, out_root


def set_seed(seed):
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def maybe_enable_mixed_precision(flag):
    if not flag:
        return
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[INFO] Mixed precision enabled (global policy = mixed_float16).")
    except Exception as e:
        print(f"[WARN] Could not enable mixed precision: {e}")


# ---- Data ----
def load_datasets(data_root, img_size, batch_size, seed):
    import tensorflow as tf

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    # Light prefetching (MacBook Air friendly)
    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    print(f"[INFO] Classes: {class_names}")
    return train_ds, val_ds, class_names


# ---- Model ----
def get_preprocess_and_backbone(backbone_name, img_size):
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

    if backbone_name == "mobilenet_v2":
        preprocess = lambda x: layers.Lambda(mv2_pre, name="mv2_pre")(x)
        backbone = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )

    elif backbone_name == "efficientnet_b0":
        preprocess = lambda x: layers.Lambda(eff_pre, name="eff_pre")(x)
        backbone = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return preprocess, backbone


def build_model(backbone_name, img_size, num_classes=2, use_augment=True):
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(img_size, img_size, 3), name="input")

    # Light augmentations (train-time only)
    x = inputs
    if use_augment:
        aug = [
            layers.RandomFlip("horizontal", name="aug_flip"),
            layers.RandomRotation(0.05, name="aug_rot"),
            layers.RandomZoom(0.1, name="aug_zoom"),
        ]
        for a in aug:
            x = a(x)

    preprocess, backbone = get_preprocess_and_backbone(backbone_name, img_size)
    x = preprocess(x)

    backbone.trainable = False  # phase 1: frozen
    x = backbone(x, training=False)  # ensure BN in inference mode while frozen

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs, outputs, name=f"{backbone_name}_catdog")
    return model, backbone


# ---- Training utils ----
def compile_model(model, lr):
    from tensorflow.keras import optimizers, losses, metrics

    opt = optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def make_callbacks(out_root):
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    best_path = out_root / "best.keras"
    ckpt = ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    return [ckpt, es, rlrop]


def unfreeze_top_layers(backbone, top_n):
    if top_n == 0:
        return
    # Freeze all first
    for layer in backbone.layers:
        layer.trainable = False
    # Unfreeze top N
    for layer in backbone.layers[-top_n:]:
        layer.trainable = True
    print(f"[INFO] Unfroze top {top_n} layers of backbone.")


def evaluate_and_log(model, val_ds):
    results = model.evaluate(val_ds, verbose=0)
    # results aligns to model.metrics_names (loss, accuracy)
    names = model.metrics_names
    return {name: float(val) for name, val in zip(names, results)}


def save_label_map(out_root, class_names):
    label_map = {name: idx for idx, name in enumerate(class_names)}
    with (out_root / "label_map.json").open("w") as f:
        json.dump(label_map, f, indent=2)
    return label_map


def save_training_log(out_root, args, metrics, class_names):
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "data": str(args.data),
            "output": str(args.output),
            "backbone": args.backbone,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "freeze_epochs": args.freeze_epochs,
            "lr": args.lr,
            "finetune_lr": args.finetune_lr,
            "finetune_top": args.finetune_top,
            "seed": args.seed,
            "use_mixed_precision": bool(args.use_mixed_precision),
        },
        "val_metrics": metrics,
        "class_names": class_names,
        "artifacts": {
            "best_model": str((Path(args.output) / "best.keras").resolve()),
            "label_map": str((Path(args.output) / "label_map.json").resolve()),
        },
    }
    with (Path(args.output) / "training_log.json").open("w") as f:
        json.dump(log, f, indent=2)


# ---- Main ----
def main():
    args = parse_args()
    data_root, out_root = validate_args(args)
    set_seed(args.seed)
    maybe_enable_mixed_precision(args.use_mixed_precision)

    print(
        "[CONFIG]\n"
        f"  data        = {data_root}\n"
        f"  output      = {out_root}\n"
        f"  backbone    = {args.backbone}\n"
        f"  img_size    = {args.img_size}\n"
        f"  batch_size  = {args.batch_size}\n"
        f"  epochs      = {args.epochs} (freeze {args.freeze_epochs} + finetune {max(0, args.epochs-args.freeze_epochs)})\n"
        f"  lr          = {args.lr}\n"
        f"  finetune_lr = {args.finetune_lr}\n"
        f"  finetune_top= {args.finetune_top}\n"
        f"  seed        = {args.seed}\n"
        f"  mp_policy   = {'mixed_float16' if args.use_mixed_precision else 'float32'}\n"
    )

    # Data
    train_ds, val_ds, class_names = load_datasets(data_root, args.img_size, args.batch_size, args.seed)
    num_classes = len(class_names)
    if num_classes != 2:
        print(f"[WARN] Expected 2 classes (cat/dog) but found: {class_names}")

    # Model
    model, backbone = build_model(args.backbone, args.img_size, num_classes=num_classes, use_augment=True)

    # Phase 1: head-only
    print("[INFO] Phase 1: training classifier head (backbone frozen).")
    compile_model(model, lr=args.lr)
    callbacks = make_callbacks(out_root)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.freeze_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: fine-tuning (optional)
    remaining = max(0, args.epochs - args.freeze_epochs)
    if remaining > 0 and args.finetune_top > 0:
        print(f"[INFO] Phase 2: fine-tuning top {args.finetune_top} layers of backbone.")
        unfreeze_top_layers(backbone, top_n=args.finetune_top)
        compile_model(model, lr=args.finetune_lr)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        print("[INFO] Skipping fine-tuning phase.")

    # Evaluate & save artifacts
    metrics = evaluate_and_log(model, val_ds)
    print(f"[RESULT] Validation metrics: {metrics}")

    save_label_map(out_root, class_names)
    save_training_log(out_root, args, metrics, class_names)

    print(f"[DONE] Best model saved to: {out_root / 'best.keras'}")
    print(f"[DONE] Label map saved to:  {out_root / 'label_map.json'}")
    print(f"[DONE] Log saved to:        {out_root / 'training_log.json'}")


if __name__ == "__main__":
    main()
