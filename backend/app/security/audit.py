"""Tamper-evident audit log.

Every security-relevant action appends an entry whose hash chains to the
previous entry (like a blockchain). The AI subsystems have no API to delete
or rewrite entries - the Database layer exposes no DELETE on audit_log.
verify_chain() re-computes the chain to detect tampering.
"""
from __future__ import annotations

import hashlib
import json

from ..database import db


def _hash_entry(seq: int, ts: float, actor: str, action: str, detail: str, prev_hash: str) -> str:
    payload = json.dumps(
        {"seq": seq, "ts": ts, "actor": actor, "action": action,
         "detail": detail, "prev_hash": prev_hash},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def log(actor: str, action: str, detail: dict | str | None = None) -> None:
    detail_s = json.dumps(detail, ensure_ascii=False, default=str) if isinstance(detail, (dict, list)) else (detail or "")
    last = db.query_one("SELECT seq, entry_hash FROM audit_log ORDER BY seq DESC LIMIT 1")
    seq = (last["seq"] + 1) if last else 1
    prev = last["entry_hash"] if last else "GENESIS"
    ts = db.now()
    entry_hash = _hash_entry(seq, ts, actor, action, detail_s, prev)
    db.execute(
        "INSERT INTO audit_log(ts, actor, action, detail, prev_hash, entry_hash) VALUES(?,?,?,?,?,?)",
        (ts, actor, action, detail_s, prev, entry_hash),
    )


def tail(limit: int = 100) -> list[dict]:
    return db.query("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))


def verify_chain() -> dict:
    rows = db.query("SELECT * FROM audit_log ORDER BY seq ASC")
    prev = "GENESIS"
    for r in rows:
        expect = _hash_entry(r["seq"], r["ts"], r["actor"], r["action"], r["detail"], prev)
        if expect != r["entry_hash"] or r["prev_hash"] != prev:
            return {"ok": False, "broken_at_seq": r["seq"], "entries": len(rows)}
        prev = r["entry_hash"]
    return {"ok": True, "entries": len(rows)}
