from __future__ import annotations

import torch
from torch import nn


class ByteLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 256, embedding_dim: int = 128, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(inputs)
        outputs, hidden = self.gru(embeddings, hidden)
        logits = self.output(outputs)
        return logits, hidden