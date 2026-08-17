"""Layered memory: short-term (decaying), episodic (events), semantic (facts).

All memories are vectorized (same hash-vectorizer as the neural core) for
cosine-similarity retrieval. The owner controls what is stored: every write
goes through MemoryStore.remember() which checks the 'learning_enabled'
control and per-kind pinning rules. Nothing is deleted silently; forgetting
in short-term memory is importance-based decay, and audit-logged.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from ..database import db
from ..models_core.neural_core import hash_vectorize
from ..security import audit

SHORT_TERM_CAP = 50
DECAY_HALF_LIFE_S = 3600 * 6  # short-term importance halves every 6h


class MemoryStore:
    def remember(self, text: str, kind: str = "episodic", importance: float = 0.5,
                 pinned: bool = False, text_hash: Optional[str] = None, source: str = "owner",
                 confidence: float = 1.0, expires_at: Optional[float] = None, approved: bool = True) -> Optional[str]:
        controls = db.get_controls()
        if not controls.get("learning_enabled", True) and kind != "short_term":
            return None
        if text_hash is None:
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()
        vec = hash_vectorize(text).tolist()
        mid = db.new_id()
        now = db.now()
        db.execute(
            "INSERT INTO memory(id, kind, text, text_hash, vector, importance, pinned, created_at, last_access, source, confidence, expires_at, approved)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, kind, text, text_hash, json.dumps(vec), importance, int(pinned), now, now, source, confidence, expires_at, int(approved)),
        )
        self._enforce_short_term_cap()
        return mid

    def _enforce_short_term_cap(self) -> None:
        rows = db.query(
            "SELECT id, importance, pinned, last_access FROM memory WHERE kind='short_term'"
        )
        if len(rows) <= SHORT_TERM_CAP:
            return
        now = db.now()

        def decayed(r):
            age = now - r["last_access"]
            return r["importance"] * (0.5 ** (age / DECAY_HALF_LIFE_S))

        rows.sort(key=lambda r: (r["pinned"], decayed(r)))
        for r in rows[: len(rows) - SHORT_TERM_CAP]:
            if r["pinned"]:
                continue
            db.execute("DELETE FROM memory WHERE id=?", (r["id"],))
            audit.log("system", "short_term_decay", {"memory_id": r["id"]})

    def recall(self, query: str, kinds: tuple = ("episodic", "semantic", "short_term"),
               top_k: int = 5) -> list[dict]:
        qv = hash_vectorize(query)
        rows = db.query(
            f"SELECT * FROM memory WHERE kind IN ({','.join('?' * len(kinds))}) AND (expires_at IS NULL OR expires_at>?) AND approved=1", tuple(kinds)+ (db.now(),)
        )
        scored = []
        for r in rows:
            v = json.loads(r["vector"])
            import torch
            sim = float(torch.nn.functional.cosine_similarity(
                qv.unsqueeze(0), torch.tensor(v).unsqueeze(0)).item())
            if math.isnan(sim):
                sim = 0.0
            scored.append((sim * (0.5 + r["importance"]), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, r in scored[:top_k]:
            db.execute("UPDATE memory SET last_access=? WHERE id=?", (db.now(), r["id"]))
            out.append({"id": r["id"], "kind": r["kind"], "text": r["text"],
                        "score": round(sim, 4), "pinned": bool(r["pinned"]), "source": r.get("source"), "confidence": r.get("confidence",1.0)})
        return out

    def list_memories(self, kind: Optional[str] = None, limit: int = 200) -> list[dict]:
        if kind:
            return db.query("SELECT id, kind, text, importance, pinned, created_at FROM memory"
                            " WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit))
        return db.query("SELECT id, kind, text, importance, pinned, created_at FROM memory"
                        " ORDER BY created_at DESC LIMIT ?", (limit,))

    def pin(self, memory_id: str, pinned: bool) -> None:
        db.execute("UPDATE memory SET pinned=? WHERE id=?", (int(pinned), memory_id))
        audit.log("owner", "memory_pin", {"id": memory_id, "pinned": pinned})

    def forget(self, memory_id: str, actor: str = "owner") -> None:
        db.execute("DELETE FROM memory WHERE id=?", (memory_id,))
        audit.log(actor, "memory_deleted", {"id": memory_id})


memory_store = MemoryStore()
