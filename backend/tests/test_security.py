"""Security invariants: the AI cannot grant itself privileges, decide its own
approvals, tamper with the audit log, or escape the sandbox."""
import pytest


def test_approval_system_actor_cannot_decide():
    from app.approval.system import approval_system
    aid = approval_system.request("config_change", {
        "change": "x", "reason": "x", "benefit": "x", "affected_files": [],
        "test_results": {}, "performance_before_after": {},
        "risks": "x", "resources_required": "x", "estimated_cost": "x",
        "new_values": {"training_enabled": False}})
    with pytest.raises(PermissionError):
        approval_system.decide(aid, "approved", actor="system")
    # owner can decide
    out = approval_system.decide(aid, "approved", actor="owner")
    assert out["status"] == "approved"


def test_no_privilege_escalation_api_exists():
    """The API surface must not expose endpoints for disabling approvals,
    changing the owner token, or deleting audit entries."""
    from app.main import app
    paths = [getattr(r, "path", "") for r in app.routes]
    forbidden = [p for p in paths if any(k in p.lower() for k in
                 ("grant", "privilege", "disable-approval", "owner-token",
                  "audit/delete", "reset-owner"))]
    assert forbidden == [], f"forbidden endpoints present: {forbidden}"


def test_audit_chain_detects_tampering():
    from app.security import audit
    from app.database import db
    audit.log("system", "e1", {"a": 1})
    audit.log("system", "e2", {"b": 2})
    assert audit.verify_chain()["ok"]
    # simulate tampering with a past entry
    db.execute("UPDATE audit_log SET detail='hacked' WHERE seq=1")
    res = audit.verify_chain()
    assert not res["ok"] and res["broken_at_seq"] == 1


def test_audit_log_has_no_delete_api():
    from app.security import audit
    assert not hasattr(audit, "delete")
    assert not hasattr(audit, "clear")


def test_bad_token_rejected(client):
    r = client.get("/api/v1/status", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 403
    r = client.get("/api/v1/status")
    assert r.status_code == 401
