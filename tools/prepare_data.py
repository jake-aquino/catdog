import argparse
import os


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





def main():
    args = parse_args()
    
    print(args)

    seed = args.seed

    splits = ['train', 'val']

    class_names = ['cat', 'dog']

    output_path = os.path.expanduser(args.output)

    create_folders(output_path, splits, class_names)




if __name__ == '__main__':
    main()