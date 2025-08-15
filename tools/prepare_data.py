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

    # Optional: preprocessing parameters
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

    # Optional: reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help= 'Seed for reproducibility'
    )

    # Optional: sample creation
    parser.add_argument(
        '--make-sample',
        action='store_true',
        help='If set, also create data/sample/ with a few images per class'
    )

    # Local rank (kept from original for multi-GPU compat, default unused here)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # Ensure LOCAL_RANK is set (even though we likely don’t use it here)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

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