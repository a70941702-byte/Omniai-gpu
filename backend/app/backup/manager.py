"""Backup & Rollback. Snapshots the DB + checkpoints into a timestamped
directory. The AI cannot delete backups (no delete API exposed)."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..config import BACKUPS_DIR, CHECKPOINTS_DIR, DB_PATH
from ..database import db
from ..security import audit


class BackupManager:
    def create(self, note: str = "") -> dict:
        bid = db.new_id()
        dest = BACKUPS_DIR / f"backup_{int(time.time())}_{bid}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DB_PATH, dest / "omniai.db")
        ckpt_dest = dest / "checkpoints"
        shutil.copytree(CHECKPOINTS_DIR, ckpt_dest, dirs_exist_ok=True)
        db.execute("INSERT INTO backups(id, path, note, created_at) VALUES(?,?,?,?)",
                   (bid, str(dest), note, db.now()))
        audit.log("system", "backup_created", {"id": bid, "note": note})
        return {"id": bid, "path": str(dest), "note": note}

    def list(self) -> list[dict]:
        return db.query("SELECT * FROM backups ORDER BY created_at DESC")

    def restore(self, backup_id: str, actor: str = "owner") -> dict:
        """Restore DB from a backup. Owner-only (called via API with owner auth).
        Restarts are required to reload state; we copy the file back."""
        row = db.query_one("SELECT * FROM backups WHERE id=?", (backup_id,))
        if not row:
            raise ValueError("backup not found")
        src = Path(row["path"]) / "omniai.db"
        if not src.exists():
            raise ValueError("backup file missing")
        shutil.copy2(src, DB_PATH)
        audit.log(actor, "backup_restored", {"id": backup_id})
        return {"restored": backup_id, "note": "restart the server to load restored state"}


backup_manager = BackupManager()
