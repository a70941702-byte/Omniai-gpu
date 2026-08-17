"""Knowledge Base + RAG.

Chunks of curated knowledge are stored as vectorized documents and retrieved
by cosine similarity. Retrieval is used for *knowledge updates* (facts change
fast); training is used for *capability changes* (skills). The Orchestrator
decides which path fits - that decision is logged per cycle.

Document ingestion pipeline: CLEAN -> FILTER -> DEDUPLICATE -> QUALITY CHECK
-> PRIVACY CHECK -> versioned store. No raw data goes straight in.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import torch

from ..database import db
from ..models_core.neural_core import hash_vectorize
from ..security import audit

_PII_PATTERNS = [
    re.compile(r"\b\d{14,16}\b"),                                   # card-like numbers
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                         # emails
    re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),  # phones
]


def _quality_score(text: str) -> float:
    words = text.split()
    if len(words) < 3:
        return 0.1
    unique_ratio = len(set(words)) / len(words)
    length_score = min(len(words) / 50.0, 1.0)
    return round(0.5 * unique_ratio + 0.5 * length_score, 3)


class KnowledgeBase:
    def ingest(self, documents: list[str], source: str = "owner") -> dict:
        """Full ingestion pipeline. Returns a quality report."""
        report = {"received": len(documents), "accepted": 0, "rejected_duplicate": 0,
                  "rejected_quality": 0, "rejected_privacy": 0}
        seen_hashes = {
            r["text_hash"] for r in db.query("SELECT DISTINCT text_hash FROM memory WHERE kind='semantic'")
        } if db.query_one("SELECT name FROM sqlite_master WHERE type='table' AND name='memory'") else set()

        for doc in documents:
            clean = re.sub(r"\s+", " ", doc).strip()
            if not clean:
                report["rejected_quality"] += 1
                continue
            h = hashlib.sha256(clean.encode()).hexdigest()
            if h in seen_hashes:
                report["rejected_duplicate"] += 1
                continue
            if any(p.search(clean) for p in _PII_PATTERNS):
                report["rejected_privacy"] += 1
                audit.log("system", "kb_reject_privacy", {"source": source})
                continue
            q = _quality_score(clean)
            if q < 0.35:
                report["rejected_quality"] += 1
                continue
            # Store as semantic memory with hash for dedup; record hash in audit
            from ..memory.store import memory_store
            doc_id=hashlib.sha256((source+clean).encode()).hexdigest()[:16]
            # Chunk before indexing so retrieval can cite the exact source segment.
            chunks=[clean[i:i+800] for i in range(0,len(clean),700)] or [clean]
            for chunk in chunks:
                memory_store.remember(chunk, kind="semantic", importance=min(0.4 + q * 0.6, 1.0),
                                      text_hash=hashlib.sha256(chunk.encode()).hexdigest(), source=source,
                                      confidence=q, approved=True)
                db.execute("INSERT INTO rag_chunks(id,document_id,chunk_index,text,source,embedding,created_at) VALUES(?,?,?,?,?,?,?)",
                           (db.new_id(),doc_id,len([x for x in chunks if x==chunk])-1,chunk,source,json.dumps(hash_vectorize(chunk).tolist()),db.now()))
            seen_hashes.add(h)
            report["accepted"] += 1

        audit.log("system", "kb_ingest", {"source": source, **report})
        return report

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        from ..memory.store import memory_store
        return memory_store.recall(query, kinds=("semantic",), top_k=top_k)


knowledge_base = KnowledgeBase()
