"""
CLI entrypoint for Zero-Leakage 5-Fold Downstream Fine-Tuning (`LP-FT` vs `Scratch`)
and Integrated Gradients XAI across all 4 SPECTRA-BREAST benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectrabreast.training import run_downstream_finetuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-Fold Zero-Leakage LP-FT Fine-Tuning & XAI.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune/smae_hw_on_riverd_oral.yaml",
        help="Path to fine-tuning YAML config.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Override pre-trained checkpoint path.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["riverd_oral_hw", "riverd_oral_ch", "ibd", "bacteria_id", "dataset_renzo"],
        help="Override downstream benchmark dataset.",
    )
    parser.add_argument("--regime", type=str, default=None, help="Override spectral regime (default/FP/CH/FR/native/broad).")
    parser.add_argument("--out-dir", type=str, default=None, help="Override experiment output directory.")
    parser.add_argument("--no-scratch", action="store_true", help="Skip the Scratch Baseline comparison.")
    parser.add_argument("--folds", type=int, default=None, help="Override number of CV folds (default: 5).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    res = run_downstream_finetuning(
        config_path=repo_root / args.config,
        repo_root=repo_root,
        checkpoint_override=args.checkpoint,
        benchmark_override=args.benchmark,
        regime_override=args.regime,
        output_dir_override=args.out_dir,
        run_scratch_override=False if args.no_scratch else None,
        n_folds_override=args.folds,
    )
    print(f"\n[DONE] Downstream results saved in: {res['output_dir']}")
    print(f"[DONE] Executed notebook saved at : {res['executed_notebook']}")


if __name__ == "__main__":
    main()
