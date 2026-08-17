"""Anti-forgetting toolkit.

Three complementary mechanisms, all actually applied during training:

1. ReplayBuffer: keeps samples from every dataset version ever trained on.
   Each new training batch mixes old samples (experience replay).
2. EWC (Elastic Weight Consolidation): after each successful training run we
   estimate the Fisher information of the parameters on old data and penalize
   moving important weights in later runs.
3. Knowledge Distillation: the candidate is trained to match the *teacher*
   (current model) soft outputs on replay data, preserving behavior.
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F

from ..config import EWC_LAMBDA, KD_ALPHA, REPLAY_RATIO
from ..models_core.neural_core import NeuralCore, hash_vectorize


class ReplayBuffer:
    """Persists across cycles (backed by the datasets table) and serves a
    mixed batch: new samples + replay of old ones at REPLAY_RATIO."""

    def __init__(self):
        self.old: list[tuple[str, int]] = []  # (text, label_idx)

    def add(self, samples: list[tuple[str, int]]) -> None:
        self.old.extend(samples)

    def mixed_batch(self, new: list[tuple[str, int]]) -> list[tuple[str, int]]:
        if not self.old:
            return list(new)
        n_replay = int(len(new) * REPLAY_RATIO / max(1e-9, 1 - REPLAY_RATIO))
        replay = random.sample(self.old, min(n_replay, len(self.old)))
        return list(new) + replay

    def __len__(self):
        return len(self.old)


class EWC:
    """Diagonal-Fisher EWC. fit() after a training run; penalty() during the
    next one. Important weights for old tasks are anchored."""

    def __init__(self):
        self.means: Optional[dict] = None
        self.fisher: Optional[dict] = None

    def fit(self, model: NeuralCore, samples: list[tuple[str, int]], n: int = 200) -> None:
        if not samples:
            return
        model.eval()
        fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        means = {name: p.detach().clone() for name, p in model.named_parameters()}
        for text, label in random.sample(samples, min(n, len(samples))):
            model.zero_grad()
            x = hash_vectorize(text, model.vocab_dim).unsqueeze(0)
            logp = F.log_softmax(model(x), dim=-1)
            loss = -logp[0, label]
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
        for name in fisher:
            fisher[name] /= max(1, min(n, len(samples)))
        self.means, self.fisher = means, fisher
        model.train()

    def penalty(self, model: NeuralCore) -> torch.Tensor:
        if self.means is None:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for name, p in model.named_parameters():
            loss = loss + (self.fisher[name] * (p - self.means[name]) ** 2).sum()
        return EWC_LAMBDA * loss


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      temperature: float = 2.0) -> torch.Tensor:
    """KL divergence between softened teacher/student outputs."""
    t = F.log_softmax(student_logits / temperature, dim=-1)
    s = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(t, s, reduction="batchmean") * (temperature ** 2) * KD_ALPHA
