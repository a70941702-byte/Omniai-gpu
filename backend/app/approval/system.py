"""Approval System - the owner is the final authority.

Autonomous work (research, experiments, training, testing, bug-fix drafts)
never requires approval. Applying anything to production (adopting a model,
changing code, changing core rules) creates a pending approval containing:
change / reason / benefit / affected files / test results / performance
before-after / risks / required resources / estimated cost.

The AI has no code path that marks its own requests approved: decide()
verifies the owner actor, and there is no API for the system actor to call
it. This property is covered by tests.
"""
from __future__ import annotations

import json
from typing import Optional

from ..database import db
from ..security import audit

REQUIRED_FIELDS = ("change", "reason", "benefit", "affected_files",
                   "test_results", "performance_before_after", "risks",
                   "resources_required", "estimated_cost")


class ApprovalSystem:
    def request(self, kind: str, payload: dict, actor: str = "system") -> str:
        missing = [f for f in REQUIRED_FIELDS if f not in payload]
        if missing:
            raise ValueError(f"approval payload missing fields: {missing}")
        aid = db.new_id()
        db.execute(
            "INSERT INTO approvals(id, kind, payload, status, created_at) VALUES(?,?,?,?,?)",
            (aid, kind, json.dumps(payload, ensure_ascii=False), "pending", db.now()),
        )
        audit.log(actor, "approval_requested", {"id": aid, "kind": kind})
        return aid

    def decide(self, approval_id: str, decision: str, note: str = "",
               actor: str = "owner") -> dict:
        if actor != "owner":
            audit.log("security", "approval_forbidden", {"by": actor, "id": approval_id})
            raise PermissionError("only the owner can decide approvals")
        if decision not in ("approved", "rejected", "more_tests"):
            raise ValueError("decision must be approved | rejected | more_tests")
        row = db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if not row:
            raise ValueError("approval not found")
        if row["status"] != "pending":
            raise ValueError(f"approval already {row['status']}")
        db.execute("UPDATE approvals SET status=?, decision_note=?, decided_at=? WHERE id=?",
                   (decision, note, db.now(), approval_id))
        audit.log("owner", "approval_decided", {"id": approval_id, "decision": decision})
        return self.get(approval_id)

    def get(self, approval_id: str) -> dict:
        row = db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        if row:
            row["payload"] = json.loads(row["payload"])
        return row

    def list(self, status: Optional[str] = None) -> list[dict]:
        if status:
            rows = db.query("SELECT * FROM approvals WHERE status=? ORDER BY created_at DESC", (status,))
        else:
            rows = db.query("SELECT * FROM approvals ORDER BY created_at DESC")
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def execute_if_approved(self, approval_id: str) -> dict:
        """Apply an approved change. This is the ONLY path to production."""
        row = self.get(approval_id)
        if not row or row["status"] != "approved":
            raise PermissionError("change is not approved")
        kind, payload = row["kind"], row["payload"]
        if kind == "model_adoption":
            from ..models_core import registry
            result = registry.adopt_model(payload["candidate_model_id"], actor="owner")
            db.execute("UPDATE approvals SET decision_note=COALESCE(decision_note,'') || ? WHERE id=?",(" | applied",approval_id))
            return {"applied": True, "model": result}
        if kind == "config_change":
            for k, v in payload["new_values"].items():
                db.set_control(k, v)
            audit.log("owner", "controls_changed", payload["new_values"])
            return {"applied": True, "controls": db.get_controls()}
        if kind == "code_change":
            from ..agents.coding_agent import coding_agent
            if not db.get_controls().get("code_editing_enabled", True):
                raise PermissionError("code editing disabled by owner")
            patch = payload.get("patch", "")
            audit.log("owner", "code_change_approved", {"files": payload.get("affected_files")})
            result = coding_agent.apply_approved_patch(patch, payload.get("affected_files", []))
            if not result.get("applied"):
                return result
            return result
        raise ValueError(f"unknown approval kind: {kind}")


approval_system = ApprovalSystem()
