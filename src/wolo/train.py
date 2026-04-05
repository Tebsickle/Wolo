from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .data import ZimByteSampler
from .model import ByteLanguageModel


@dataclass
class TrainingConfig:
    zim_path: Path = Path("data/wiki_en_all_maxi_2026-02.zim")
    min_entry_id: int | None = None
    max_entry_id: int | None = None
    sequence_length: int = 192
    batch_size: int = 8
    steps: int = 500
    learning_rate: float = 3e-4
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    checkpoint_dir: Path = Path("checkpoints")
    metrics_path: Path = Path("runs/metrics.jsonl")
    log_every: int = 50
    checkpoint_every: int = 250
    seed: int = 0


def _save_metrics(metrics_path: Path, payload: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_checkpoint(checkpoint_dir: Path, step: int, config: TrainingConfig, model: ByteLanguageModel, optimizer: torch.optim.Optimizer) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "config": {
                "embedding_dim": config.embedding_dim,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
            },
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_training(config: TrainingConfig, resume_from: Path | None = None) -> None:
    torch.manual_seed(config.seed)
    device = torch.device("cpu")

    sampler = ZimByteSampler(
        zim_path=config.zim_path,
        min_entry_id=config.min_entry_id,
        max_entry_id=config.max_entry_id,
        seed=config.seed,
    )

    model = ByteLanguageModel(
        vocab_size=256,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_step = 1
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step"] + 1
        print(f"Resumed from checkpoint at step {checkpoint['step']}, continuing from step {start_step}")

    started_at = time.perf_counter()

    try:
        for step in range(start_step, config.steps + 1):
            batch_inputs, batch_targets = sampler.sample_batch(config.batch_size, config.sequence_length)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_inputs)
            loss = criterion(logits.reshape(-1, 256), batch_targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % config.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - started_at
                metrics = {
                    "step": step,
                    "loss": float(loss.item()),
                    "elapsed_seconds": round(elapsed, 2),
                    "sequence_length": config.sequence_length,
                    "batch_size": config.batch_size,
                }
                print(json.dumps(metrics))
                _save_metrics(config.metrics_path, metrics)

            if step % config.checkpoint_every == 0 or step == config.steps:
                checkpoint_path = _save_checkpoint(config.checkpoint_dir, step, config, model, optimizer)
                print(json.dumps({"checkpoint": str(checkpoint_path)}))
    finally:
        sampler.close()