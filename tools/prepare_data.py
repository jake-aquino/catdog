import argparse
import os
import sys
import json
import shutil
import random
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

from PIL import Image

CLASSES = ["cat", "dog"]
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




def main():
    args = parse_args()

    input_root, output_root = validate_args(args)

    print(args)

    seed = args.seed

    splits = ['train', 'val']

    class_names = ['cat', 'dog']

    output_path = os.path.expanduser(args.output)

    create_folders(output_path, splits, class_names)
    
    class_paths = collect_images(input_root, CLASSES)

    splits = get_splits(class_paths, args.val, args.test, args.seed)






if __name__ == '__main__':
    main()