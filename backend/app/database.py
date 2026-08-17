"""SQLite storage layer. One table per subsystem; all writes go through here
so the audit log can observe state changes. Thread-safe via a lock."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import DB_PATH


class Database:
    def __init__(self, path: Path = DB_PATH):
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,          -- episodic | semantic | short_term
                    text TEXT NOT NULL,
                    text_hash TEXT,            -- sha256 for dedup (KB ingestion)
                    vector TEXT NOT NULL,      -- JSON list of floats
                    importance REAL NOT NULL DEFAULT 0.5,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,      -- raw | cleaned | approved | rejected
                    samples TEXT NOT NULL,     -- JSON list
                    quality_report TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(name, version)
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    parent_id TEXT,
                    intent_labels TEXT,        -- JSON list of intent names
                    config TEXT,               -- training configuration JSON
                    status TEXT NOT NULL,      -- candidate | current | best_historical | archived | rejected
                    metrics TEXT,              -- evaluation results JSON
                    resource_usage TEXT,
                    checkpoint_path TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_cycles (
                    id TEXT PRIMARY KEY,
                    cycle_no INTEGER NOT NULL,
                    phase TEXT NOT NULL,       -- learn/collect/clean/evaluate/train/test/compare/protect/checkpoint/deploy/analyze
                    status TEXT NOT NULL,      -- running | deployed | rejected | failed
                    base_model_id TEXT,
                    candidate_model_id TEXT,
                    report TEXT,
                    created_at REAL NOT NULL,
                    finished_at REAL
                );
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    suite TEXT NOT NULL,
                    score REAL NOT NULL,
                    details TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,        -- model_adoption | code_change | config_change
                    payload TEXT NOT NULL,     -- the change proposal (change/reason/benefit/files/tests/perf/risks/cost)
                    status TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | more_tests
                    decision_note TEXT,
                    created_at REAL NOT NULL,
                    decided_at REAL
                );
                CREATE TABLE IF NOT EXISTS controls (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS improvement_proposals (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    analysis TEXT NOT NULL,
                    patch TEXT,
                    test_output TEXT,
                    status TEXT NOT NULL DEFAULT 'proposed',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backups (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    note TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, role TEXT NOT NULL, device_id TEXT, created_at REAL NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS tool_audit (id TEXT PRIMARY KEY, tool TEXT NOT NULL, actor TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL, ok INTEGER NOT NULL, detail TEXT);
                CREATE TABLE IF NOT EXISTS costs (id TEXT PRIMARY KEY, category TEXT NOT NULL, amount REAL NOT NULL, metadata TEXT, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS model_artifacts (id TEXT PRIMARY KEY, model_id TEXT NOT NULL, base_model TEXT, adapter TEXT, quantization TEXT, dataset_version TEXT, training_config TEXT, evaluation TEXT, parent_model TEXT, created_at REAL NOT NULL, approved_at REAL);
                CREATE TABLE IF NOT EXISTS training_samples (id TEXT PRIMARY KEY, text TEXT NOT NULL, intent TEXT, source TEXT NOT NULL, timestamp REAL NOT NULL, quality_score REAL NOT NULL, confidence REAL NOT NULL, provenance TEXT, approval_status TEXT NOT NULL); CREATE TABLE IF NOT EXISTS rag_chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, source TEXT, embedding TEXT NOT NULL, created_at REAL NOT NULL);
            """)
            for col, typ in (("source","TEXT"),("confidence","REAL"),("expires_at","REAL"),("approved","INTEGER"),("model_type","TEXT"),("base_model","TEXT"),("adapter","TEXT"),("quantization","TEXT"),("training_dataset","TEXT"),("training_config","TEXT"),("evaluation_score","REAL"),("approved_at","REAL")):
                try: self._conn.execute(f"ALTER TABLE models ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError: pass
            for col, typ in (("source","TEXT"),("confidence","REAL"),("expires_at","REAL"),("approved","INTEGER")):
                try: self._conn.execute(f"ALTER TABLE memory ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError: pass
            self._conn.execute("UPDATE memory SET source=COALESCE(source,'owner'), confidence=COALESCE(confidence,1.0), approved=COALESCE(approved,1)")
            self._conn.commit()

    # ---- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- controls ---------------------------------------------------------
    DEFAULT_CONTROLS = {
        "training_enabled": True,
        "learning_enabled": True,
        "internet_enabled": False,
        "external_models_enabled": False,
        "code_edit_enabled": True,
        "install_deps_enabled": False,
        "server_enabled": False,
        "cpu_limit_percent": 80,
        "ram_limit_mb": 2048,
        "gpu_enabled": False,
        "storage_limit_mb": 4096,
        "budget_credits": 1000,
        "budget_spent": 0,
        "autonomous_cycles": True,
        "python_enabled": True, "terminal_enabled": False, "web_enabled": False,
        "tools_enabled": True, "agent_approved_tools": False, "code_editing_enabled": True,
        "kill_switch": False, "domain_allowlist": [], "domain_blocklist": [],
        "session_ttl_s": 86400,
        "approval_read": "LOW", "approval_analyze": "LOW", "approval_tests": "MEDIUM",
        "approval_patch": "MEDIUM", "approval_code_edit": "HIGH", "approval_dependency": "HIGH",
        "approval_training": "HIGH", "approval_internet": "CRITICAL", "approval_model_promote": "CRITICAL",
    }

    def get_controls(self) -> dict:
        rows = self.query("SELECT key, value FROM controls")
        controls = dict(self.DEFAULT_CONTROLS)
        for r in rows:
            try:
                controls[r["key"]] = json.loads(r["value"])
            except Exception:
                controls[r["key"]] = r["value"]
        return controls

    def set_control(self, key: str, value: Any) -> None:
        if key not in self.DEFAULT_CONTROLS:
            raise KeyError(f"unknown control: {key}")
        self.execute(
            "INSERT INTO controls(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        aliases={"code_edit_enabled":"code_editing_enabled","code_editing_enabled":"code_edit_enabled"}
        if key in aliases:
            self.execute("INSERT INTO controls(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(aliases[key],json.dumps(value)))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    @staticmethod
    def now() -> float:
        return time.time()


db = Database()
