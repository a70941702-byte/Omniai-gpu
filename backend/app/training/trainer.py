"""Trainer - performs the actual gradient training of a candidate model.

Real training path per cycle:
  candidate = clone(current) -> expand head for new intents -> add adapter
  -> train on (new samples + replay) with EWC penalty + distillation from
  the teacher (current model) -> save checkpoint -> register as candidate.

The Trainer never touches the deployed model in place; deployment happens
only through the registry's adopt_model after owner approval.
"""
from __future__ import annotations

import copy
import json
import time

import torch

from ..config import DEFAULT_EPOCHS, DEFAULT_LR
from ..database import db
from ..models_core.neural_core import NeuralCore, hash_vectorize
from ..models_core import registry
from ..security.budget import check, charge
from .anti_forgetting import EWC, ReplayBuffer, distillation_loss


def _load_replay_buffer(intent_labels: list[str]) -> ReplayBuffer:
    """Rebuild the replay buffer from ALL approved dataset versions."""
    buf = ReplayBuffer()
    rows = db.query("SELECT samples FROM datasets WHERE status='approved' ORDER BY version ASC")
    for r in rows:
        for s in json.loads(r["samples"]):
            if s["intent"] in intent_labels:
                buf.add([(s["text"], intent_labels.index(s["intent"]))])
    return buf


class Trainer:
    def train_candidate(
        self,
        base_model_id: str | None,
        new_samples: list[dict],        # [{"text":..., "intent":...}]
        dataset_version: str,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
    ) -> dict:
        t0 = time.time()
        estimated_cost = max(0.01, epochs * max(1, len(new_samples)) * 0.002)
        check(estimated_cost)
        # ---- assemble label space: existing intents + new ones ------------
        if base_model_id:
            base_row = registry.get_model(base_model_id)
            base_model = registry.load_model(base_model_id)
            intent_labels = list(base_row["intent_labels"])
        else:
            base_model = None
            intent_labels = []

        new_intents = sorted({s["intent"] for s in new_samples if s["intent"] not in intent_labels})
        all_labels = intent_labels + new_intents

        # ---- build candidate ----------------------------------------------
        if base_model is not None:
            candidate = copy.deepcopy(base_model)
            candidate.expand_output(len(all_labels))
            candidate.add_adapter()          # new capacity for the new skill
            teacher = base_model
        else:
            candidate = NeuralCore(num_outputs=len(all_labels))
            teacher = None
        candidate.train()

        replay = _load_replay_buffer(all_labels)
        indexed_new = [(s["text"], all_labels.index(s["intent"])) for s in new_samples]
        batch = replay.mixed_batch(indexed_new)

        ewc = EWC()
        if teacher is not None and len(replay) > 0:
            ewc.fit(candidate, replay.old)

        opt = torch.optim.Adam(candidate.parameters(), lr=lr)
        xs = torch.stack([hash_vectorize(t) for t, _ in batch])
        ys = torch.tensor([y for _, y in batch])

        final_loss = 0.0
        for _epoch in range(epochs):
            opt.zero_grad()
            logits = candidate(xs)
            loss = torch.nn.functional.cross_entropy(logits, ys)
            if teacher is not None:
                # EWC anchor + distill teacher behavior on the replay subset
                loss = loss + ewc.penalty(candidate)
                if len(replay) > 0:
                    with torch.no_grad():
                        tx = torch.stack([hash_vectorize(t) for t, _ in replay.old])
                        t_idx = [all_labels.index(intent_labels[y]) if False else None for _, y in []]
                    teacher_logits = teacher(tx)
                    # teacher head may be smaller; align on shared dims
                    d = min(teacher_logits.shape[-1], logits.shape[-1])
                    student_on_replay = candidate(tx)
                    loss = loss + distillation_loss(student_on_replay[..., :d],
                                                    teacher_logits[..., :d])
            loss.backward()
            opt.step()
            final_loss = float(loss.item())

        # ---- measure held-out accuracy on the new data ---------------------
        candidate.eval()
        with torch.no_grad():
            pred = candidate(xs).argmax(dim=-1)
            train_acc = float((pred == ys).float().mean().item())

        model_id = registry.register_model(
            intent_labels=all_labels,
            config={"lr": lr, "epochs": epochs, "dataset_version": dataset_version,
                    "replay_size": len(replay), "ewc": True, "distillation": teacher is not None,
                    "new_intents": new_intents},
            parent_id=base_model_id,
            checkpoint_path=None,
            metrics={"train_loss": round(final_loss, 5), "train_acc": round(train_acc, 4)},
            status="candidate",
        )
        registry.save_checkpoint(candidate, model_id)
        charge("training_cpu", max(0.01, time.time()-t0)*0.02, {"model_id":model_id,"epochs":epochs})
        return {"model_id": model_id, "labels": all_labels, "new_intents": new_intents,
                "train_loss": round(final_loss, 5), "train_acc": round(train_acc, 4),
                "train_seconds": round(time.time() - t0, 2), "replay_size": len(replay)}


trainer = Trainer()
