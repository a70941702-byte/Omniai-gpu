"""NeuralCore - the trainable local model.

A compact hash-vectorizer + MLP with per-version Adapters. This is real
gradient-based training (backprop on CPU). On a GPU server, the same
Trainer interface can fine-tune larger models with LoRA; the registry and
adoption gates treat them identically.

Key anti-forgetting design: new capabilities extend the output head and add a
new Adapter rather than overwriting shared weights, and training mixes replay
samples + EWC penalty (see training/anti_forgetting.py).
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

import torch
import torch.nn as nn

from ..config import HIDDEN_DIM, VOCAB_DIM

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u0600-\u06FF]+")


def hash_vectorize(text: str, dim: int = VOCAB_DIM) -> torch.Tensor:
    """Deterministic bag-of-hashes vector (no external tokenizer needed)."""
    vec = torch.zeros(dim)
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = vec.norm()
    return vec / norm if norm > 0 else vec


class Adapter(nn.Module):
    """Small bottleneck adapter; each training cycle may add one."""

    def __init__(self, dim: int, bottleneck: int = 16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.relu(self.down(x)))


class NeuralCore(nn.Module):
    def __init__(self, num_outputs: int, vocab_dim: int = VOCAB_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.vocab_dim = vocab_dim
        self.hidden = hidden
        self.num_outputs = num_outputs
        self.inp = nn.Linear(vocab_dim, hidden)
        self.adapters = nn.ModuleList([Adapter(hidden)])
        self.out = nn.Linear(hidden, num_outputs)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.inp(x))
        for a in self.adapters:
            h = torch.relu(a(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(x))

    def add_adapter(self) -> None:
        self.adapters.append(Adapter(self.hidden))

    def expand_output(self, new_size: int) -> None:
        """Grow the head for new intents, preserving existing weights (no forgetting)."""
        if new_size <= self.num_outputs:
            return
        old = self.out
        new = nn.Linear(self.hidden, new_size)
        with torch.no_grad():
            new.weight[: self.num_outputs] = old.weight
            new.bias[: self.num_outputs] = old.bias
        self.out = new
        self.num_outputs = new_size

    def predict_proba(self, texts: Iterable[str]) -> torch.Tensor:
        xs = torch.stack([hash_vectorize(t, self.vocab_dim) for t in texts])
        with torch.no_grad():
            logits = self.forward(xs)
        return torch.softmax(logits, dim=-1)

    # ---- checkpointing ----------------------------------------------------
    def save(self, path) -> None:
        torch.save(
            {"state_dict": self.state_dict(),
             "num_outputs": self.num_outputs,
             "vocab_dim": self.vocab_dim,
             "hidden": self.hidden,
             "n_adapters": len(self.adapters)},
            str(path),
        )

    @classmethod
    def load(cls, path) -> "NeuralCore":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(ckpt["num_outputs"], ckpt["vocab_dim"], ckpt["hidden"])
        for _ in range(ckpt["n_adapters"] - 1):
            model.add_adapter()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
