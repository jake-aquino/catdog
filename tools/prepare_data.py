import argparse
import os
import sys
import json
import shutil
import random
from pathlib import Path

from PIL import Image

CLASSES = ["cat", "dog"]
SPLITS = ["val", "train", "test"]
IMG_FORMAT = ".jpg"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare the cats vs dogs dataset for training/testing'
    )

    # Required: input and output directories
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw dataset folder (e.g., data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save processed dataset (e.g., data/processed)'
    )

    #preprocessing parameters
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Target size (pixels) for resizing images (default: 224)'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Fraction of images to use for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Fraction of images to use for test (default: 0.15)'
    )

    #reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help= 'Seed for reproducibility'
    )

    #sample creation
    parser.add_argument(
        '--make-sample',
        action='store_true',
        help='If set, also create data/sample/ with a few images per class'
    )
    
    #images per sample
    parser.add_argument(
        "--sample-per-class",
        type=int, default=2,
        help="Sample images per class (default: 2)"
    )

    parser.add_argument(
    '--force',
    action='store_true',
    help='If set, wipe the output directory before writing'
    )

    #Local rank (kept from original for multi-GPU compat, default unused here)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def validate_args(args):
    input_root = Path(os.path.expanduser(args.input)).resolve()
    output_root = Path(os.path.expanduser(args.output)).resolve()

    if not input_root.exists() or not input_root.is_dir():
        sys.exit(f"[ERROR] Input directory does not exist or is not a directory: {input_root}")

    if args.img_size <= 0:
        sys.exit("[ERROR] --img-size must be a positive integer.")

    if not (0.0 <= args.val < 1.0) or not (0.0 <= args.test < 1.0):
        sys.exit("[ERROR] --val and --test must be in [0.0, 1.0).")

    if args.val + args.test >= 0.999999:
        sys.exit("[ERROR] --val + --test must be < 1.0 (leave room for train).")

    # Output directory handling
    if output_root.exists():
        if args.force:
            print(f"[INFO] --force given: removing existing output directory: {output_root}")
            shutil.rmtree(output_root)
        else:
            print(f"[WARN] Output directory exists and will be reused: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    return input_root, output_root

def create_folders(data_root, splits, class_names):
    for split in splits:
        split_path = os.path.join(data_root, split)
        os.makedirs(split_path, exist_ok=True)

        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)

    return

def set_seed(seed):
    random.seed(seed)

def collect_images(input_root, classes):
    class_to_paths = {}

    for c in classes:
        c_dir = input_root / c

        imgs = [img for img in c_dir.rglob("*") if img.suffix.lower() == IMG_FORMAT]
        class_to_paths[c] = sorted(imgs)
        print(f'[INFO] Class {c}: found {len(imgs)}')
        
        if len(imgs) == 0:
            print(f'[ERROR No images found for {c} in directory {c_dir}')

    return class_to_paths

def get_splits(class_to_paths, val_frac, test_frac, seed):
    rng = random.Random(seed)

    splits = {'train': {}, 'val': {}, 'test': {}}

    for c, paths in class_to_paths.items():
        buf = list(paths)
        rng.shuffle(buf)
        n = len(buf)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        n_train = n - n_val - n_test

        if n_train <= 0:
            sys.exit(f"[ERROR] Not enough images in class: {c} after split")

        splits['val'][c] = buf[:n_val]
        splits['test'][c] = buf[n_val:n_val+n_test]
        splits['train'][c] = buf[n_val+n_test:]
        
        print(f"[INFO] Split {c}: train={len(splits['train'][c])}, "
              f"val={len(splits['val'][c])}, test={len(splits['test'][c])}")

    return splits

def process_imgs(src, dst, img_size):


    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size))

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        dst = os.path.splitext(dst)[0] + IMG_FORMAT

        im.save(dst, format="JPEG", quality=92, optimize=True)

    return dst

def materialize_splits(output_root, split_dict, img_size):

    total = 0
    errors = []

    splits = split_dict.keys()

    # iterate in a fixed order
    for split_name in splits:
        per_class = split_dict[split_name]

        if split_name == "test" and sum(len(v) for v in per_class.values()) == 0:
            continue  # skip empty test

        print(f"[INFO] Writing split: {split_name}")

        for c in CLASSES:
            paths = per_class.get(c, [])
            out_dir = os.path.join(str(output_root), split_name, c)

            processed = 0
            for i, src_path in enumerate(paths, start=1):
                try:
                    dst_path = os.path.join(out_dir, os.path.basename(str(src_path)))
                    process_imgs(str(src_path), dst_path, img_size)
                    processed += 1
                    total += 1
                    if i % 500 == 0:
                        print(f"  [{split_name}/{c}] processed {i}/{len(paths)} ...")
                except Exception as e:
                    msg = f"[ERROR] {src_path} → {e}"
                    print(msg)
                    errors.append(msg)

            print(f"  [{split_name}/{c}] done: {processed} images")

    return total, errors

def make_sample_from_train(output_root, per_class):

    sample_root = Path("data") / "sample"

    if sample_root.exists():
        shutil.rmtree(sample_root)
    sample_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for c in CLASSES:
        src_dir = output_root / "train" / c
        srcs = sorted(src_dir.glob("*.jpg"))[:per_class]
        dst_dir = sample_root / c
        dst_dir.mkdir(parents=True, exist_ok=True)
        for s in srcs:
            shutil.copy2(s, dst_dir / s.name)
            copied += 1
    print(f"[INFO] Created sample set: {sample_root} (total files: {copied})")


def write_meta(output_root, args):

    meta = {
        "classes": CLASSES,
        "img_size": args.img_size,
        "val_frac": args.val,
        "test_frac": args.test,
        "seed": args.seed,
        "src_input_path": str(Path(os.path.expanduser(args.input)).resolve()),
    }

    meta_path = output_root / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Wrote meta: {meta_path}")



def main():
    args = parse_args()

    input_root, output_root = validate_args(args)

    print(
        f"[CONFIG]\n"
        f"  input = {input_root}\n"
        f"  output = {output_root}\n"
        f"  classes = {CLASSES}\n"
        f"  img_size = {args.img_size}\n"
        f"  val = {args.val} | test = {args.test}\n"
        f"  seed = {args.seed}\n"
        f"  make_sample = {args.make_sample} (per_class={args.sample_per_class})\n"
        )

    set_seed(args.seed)


    create_folders(output_root, SPLITS, CLASSES)
    
    class_paths = collect_images(input_root, CLASSES)

    split_dict = get_splits(class_paths, args.val, args.test, args.seed)

    total, erros = materialize_splits(output_root, split_dict, args.img_size)

    print(f"[INFO] Total processed images: {total}")

    write_meta(output_root, args)

    if args.make_sample:
        make_sample_from_train(output_root, args.sample_per_class)
    
        if errors:
            reports_dir = output_root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            err_file = reports_dir / "prepare_errors.txt"
            with err_file.open("w") as f:
                f.write("\n".join(errors))
            print(f"[WARN] Encountered {len(errors)} problematic files. See {err_file}")

    print("[DONE] Data preparation complete.")







if __name__ == '__main__':
    main()