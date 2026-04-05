from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wolo.model import ByteLanguageModel


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def load_model(checkpoint_path: Path) -> ByteLanguageModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model = ByteLanguageModel(
        vocab_size=256,
        embedding_dim=int(config.get("embedding_dim", 64)),
        hidden_size=int(config.get("hidden_size", 128)),
        num_layers=int(config.get("num_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_k > 0:
        k = min(top_k, probs.shape[-1])
        top_probs, top_indices = torch.topk(probs, k)
        chosen = torch.multinomial(top_probs, num_samples=1)
        return int(top_indices[chosen].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def generate(model: ByteLanguageModel, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    prompt_bytes = list(prompt.encode("utf-8", errors="ignore"))
    if not prompt_bytes:
        return ""

    with torch.no_grad():
        input_ids = torch.tensor(prompt_bytes, dtype=torch.long).unsqueeze(0)
        logits, hidden = model(input_ids)
        last_logits = logits[:, -1, :].squeeze(0)

        generated = prompt_bytes.copy()
        for _ in range(max_new_tokens):
            next_token = sample_next_token(last_logits, temperature, top_k)
            generated.append(next_token)

            next_input = torch.tensor([[next_token]], dtype=torch.long)
            logits, hidden = model(next_input, hidden)
            last_logits = logits[:, -1, :].squeeze(0)

    return bytes(generated).decode("utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interact with a trained Wolo checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load (defaults to latest in checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory containing step_*.pt checkpoints")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint is not None else find_latest_checkpoint(args.checkpoint_dir)
    model = load_model(checkpoint_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print("Enter a prompt and press Enter. Use /exit to quit.")

    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        output = generate(
            model=model,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        continuation = output[len(prompt):]
        print(f"\n{continuation}\n")


if __name__ == "__main__":
    main()
