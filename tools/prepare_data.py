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
        h
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


def main():
    return 0


if __name__ == '__main__':
    main()