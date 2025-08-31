import argparse
import os
import sys
import json
import shutil
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train model on cats vs. dogs dataset, assumes no base model exists yet'
    )

    # Required: processed data
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to pre-processed data'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/",
        help="Directory to save trained models and logs (default: models/)."
    )

    # Model / training setup
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mobilenet_v2", "efficientnet_b0"],
        default="mobilenet_v2",
        help="Base CNN to use for transfer learning (default: mobilenet_v2)."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (square) for training/inference (default: 224)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Total number of epochs (default: 12)."
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=4,
        help="Epochs to train classifier head with backbone frozen (default: 4)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for head training (default: 1e-3)."
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning backbone (default: 1e-4)."
    )
    parser.add_argument(
        "--finetune-top",
        type=int,
        default=50,
        help="Number of top layers to unfreeze for fine-tuning (default: 50)."
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Enable mixed precision training (use only if tensorflow-metal installed)."
    )

    # Legacy compatibility (not used in single-process local run)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # Keep LOCAL_RANK set, even if unused
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


