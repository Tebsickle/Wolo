from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wolo.train import TrainingConfig, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a byte-level Wikipedia language model from a ZIM archive")
    parser.add_argument("--zim-path", type=Path, default=Path("data/wiki_en_all_maxi_2026-02.zim"))
    parser.add_argument("--min-entry-id", type=int, default=None)
    parser.add_argument("--max-entry-id", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=96, help="Sequence length (default: 96 for RPi 4B)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1 for RPi 4B)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension (default: 64 for RPi 4B)")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size (default: 128 for RPi 4B)")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers (default: 1 for RPi 4B)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--metrics-path", type=Path, default=Path("runs/metrics.jsonl"))
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume training from a checkpoint (auto-detects latest if not specified)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh training regardless of existing checkpoints")
    return parser


def checkpoint_architecture_matches(checkpoint_path: Path, config: TrainingConfig) -> bool:
    """Check if checkpoint architecture matches current config."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("config", {})
        
        # Compare key architecture parameters
        if (state.get("embedding_dim") != config.embedding_dim or
            state.get("hidden_size") != config.hidden_size or
            state.get("num_layers") != config.num_layers):
            return False
        return True
    except Exception as e:
        print(f"Warning: Could not inspect checkpoint {checkpoint_path}: {e}")
        return False


def main() -> None:
    args = build_parser().parse_args()
    
    config = TrainingConfig(
        zim_path=args.zim_path,
        min_entry_id=args.min_entry_id,
        max_entry_id=args.max_entry_id,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        checkpoint_dir=args.checkpoint_dir,
        metrics_path=args.metrics_path,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    
    # Auto-detect latest checkpoint if not explicitly specified and not --fresh
    resume_from = args.resume_from
    if not args.fresh and resume_from is None and args.checkpoint_dir.exists():
        checkpoints = list(args.checkpoint_dir.glob("step_*.pt"))
        if checkpoints:
            # Extract step numbers and find the latest
            step_numbers = []
            for cp in checkpoints:
                try:
                    step = int(cp.stem.split("_")[1])
                    step_numbers.append((step, cp))
                except (IndexError, ValueError):
                    pass
            if step_numbers:
                latest_step, latest_checkpoint = max(step_numbers)
                # Check if architecture matches before auto-resuming
                if checkpoint_architecture_matches(latest_checkpoint, config):
                    resume_from = latest_checkpoint
                    print(f"Auto-detected latest checkpoint: {resume_from} (step {latest_step})")
                else:
                    print(f"Found checkpoint at step {latest_step}, but architecture mismatch detected.")
                    print(f"Starting fresh training. Use '--resume-from {latest_checkpoint}' to force load, or '--fresh' to suppress this message.")
    
    run_training(config, resume_from=resume_from)


if __name__ == "__main__":
    main()
