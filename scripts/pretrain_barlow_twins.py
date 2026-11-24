"""CLI entrypoint to run 1D-ResNet-34 (Barlow Twins) PyTorch SSL Pre-Training on an HPC GPU node."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectrabreast.training import train_barlow_twins_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train 1D-ResNet-34 Barlow Twins (PyTorch) from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/barlow_twins_all_1000.yaml",
        help="Path to Barlow Twins YAML config file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch instead of resuming from last_checkpoint.pth.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for total number of training epochs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional custom experiment output directory (prevents overwriting previous runs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override (e.g. 512 or 1024 on A100).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    res = train_barlow_twins_from_config(
        config_path=repo_root / args.config,
        repo_root=repo_root,
        resume=not args.no_resume,
        max_epochs_override=args.epochs,
        output_dir_override=args.out_dir,
        batch_size_override=args.batch_size,
    )
    print(f"\n[DONE] Best Val Barlow Twins Loss: {res['best_val_loss']:.4f}")
    print(f"[DONE] Best ResNet Encoder Checkpoint: {res['best_encoder_checkpoint']}")


if __name__ == "__main__":
    main()
