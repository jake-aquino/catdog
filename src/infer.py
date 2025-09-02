import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.models import load_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a trained cats-vs-dogs model."
    )

    p.add_argument(
        "--model",
        type=str,
        default="models/best.keras",
        help="Path to trained Keras model (default: models/best.keras)",
    )

    p.add_argument(
        "--labels",
        type=str,
        default="models/label_map.json",
        help="Path to label map JSON (default: models/label_map.json)",
    )

    p.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image file(s) to classify",
    )

    p.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only (one object per line).",
    )
    return p.parse_args()

def load_artifacts(model_path: Path, labels_path: Path):
    if not model_path.exists():
        sys.exit(f"[ERROR] Model file not found: {model_path}")
    if not labels_path.exists():
        sys.exit(f"[ERROR] Label map not found: {labels_path}")

    # Try MobilenetV2 first, then EfficientNet
    try:
        model = load_model(model_path, custom_objects={"preprocess_input": mv2_pre})
    except Exception:
        model = load_model(model_path, custom_objects={"preprocess_input": eff_pre})

    with labels_path.open("r") as f:
        label_map = json.load(f)
    id_to_name = {v: k for k, v in label_map.items()}
    return model, id_to_name


def get_input_size(model) -> int:
    # Expect shape: (None, H, W, 3)
    ishape = model.input_shape
    if isinstance(ishape, list):  # some models have multi-input; use first
        ishape = ishape[0]
    h, w = ishape[1], ishape[2]
    if not (isinstance(h, int) and isinstance(w, int) and h == w):
        # fallback to 224 if something odd
        return 224
    return int(h)


def load_and_prepare(img_path: Path, target_size: int):
    try:
        img = Image.open(img_path).convert("RGB").resize((target_size, target_size))
    except Exception as e:
        raise RuntimeError(f"Failed to open/convert image '{img_path}': {e}")

    arr = img_to_array(img)          # float32, shape (H, W, 3)
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)

    return arr


def predict_one(model, id_to_name, img_path: Path, target_size: int):
    x = load_and_prepare(img_path, target_size)
    preds = model.predict(x, verbose=0)   # shape (1, num_classes)
    probs = preds[0].astype(float)
    idx = int(np.argmax(probs))
    return {
        "image": str(img_path),
        "label_id": idx,
        "label": id_to_name.get(idx, str(idx)),
        "probability": float(probs[idx]),
        "probs": {id_to_name.get(i, str(i)): float(p) for i, p in enumerate(probs)},
    }


def main():
    args = parse_args()
    model_path = Path(os.path.expanduser(args.model)).resolve()
    labels_path = Path(os.path.expanduser(args.labels)).resolve()
    model, id_to_name = load_artifacts(model_path, labels_path)
    target_size = get_input_size(model)

    errors = 0
    for img in args.images:
        img_path = Path(img).expanduser().resolve()
        if not img_path.exists():
            print(json.dumps({"image": str(img_path), "error": "file-not-found"}) if args.json
                  else f"[ERROR] File not found: {img_path}")
            errors += 1
            continue
        try:
            out = predict_one(model, id_to_name, img_path, target_size)
            if args.json:
                print(json.dumps(out))
            else:
                print(f"{out['image']}: {out['label']}  (p={out['probability']:.3f})")
        except Exception as e:
            if args.json:
                print(json.dumps({"image": str(img_path), "error": str(e)}))
            else:
                print(f"[ERROR] {img_path}: {e}")
            errors += 1

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
