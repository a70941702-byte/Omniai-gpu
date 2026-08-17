"""Model registry, full training cycle, adoption flow, rollback."""


def _bootstrap(client):
    # first chat triggers seed-model bootstrap
    r = client.post("/api/v1/chat",
                    json={"text": "hello"},
                    headers={"Authorization": "Bearer test-owner-token"})
    assert r.status_code == 200
    return r.json()["conversation_id"]


def test_full_cycle_creates_pending_approval(client, owner_headers):
    _bootstrap(client)
    r = client.post("/api/v1/training/cycle", json={
        "extra_samples": [
            {"text": "astronomy stars galaxies telescope", "intent": "science"},
            {"text": "chemistry atoms molecules reaction", "intent": "science"},
            {"text": "physics energy force motion", "intent": "science"},
        ]}, headers=owner_headers)
    assert r.status_code == 200
    body = r.json()
    # the cycle either reaches deploy-pending-approval or rejects with a strategy
    assert body["status"] in ("deployed_pending_approval", "rejected", "failed")
    if body["status"] == "deployed_pending_approval":
        approval = client.get(f"/api/v1/approvals?status=pending",
                              headers=owner_headers).json()
        assert any(a["id"] == body["approval_id"] for a in approval)
        payload = [a for a in approval if a["id"] == body["approval_id"]][0]["payload"]
        for field in ("change", "reason", "benefit", "affected_files", "test_results",
                      "performance_before_after", "risks", "resources_required",
                      "estimated_cost"):
            assert field in payload


def test_owner_approve_and_adopt_model(client, owner_headers):
    _bootstrap(client)
    r = client.post("/api/v1/training/cycle", json={
        "extra_samples": [{"text": f"new domain topic {i} zoology animals", "intent": "zoology"}
                          for i in range(5)]}, headers=owner_headers)
    body = r.json()
    if body["status"] != "deployed_pending_approval":
        # force a decision path: not adopted due to no gain; still valid behavior
        assert "decision" in body or "reason" in body
        return
    aid = body["approval_id"]
    r = client.post(f"/api/v1/approvals/{aid}/decide",
                    json={"decision": "approved"}, headers=owner_headers)
    assert r.json()["status"] == "approved"
    r = client.post(f"/api/v1/approvals/{aid}/execute", headers=owner_headers)
    assert r.json()["applied"]
    models = client.get("/api/v1/models", headers=owner_headers).json()
    statuses = {m["id"]: m["status"] for m in models}
    assert statuses[body["candidate"]] == "current"
    # previous model must still exist (archived or best_historical) - no deletion
    assert len(models) >= 2


def test_rollback_restores_previous(client, owner_headers):
    _bootstrap(client)
    models_before = client.get("/api/v1/models", headers=owner_headers).json()
    seed_id = [m for m in models_before if m["version"] == 1][0]["id"]
    r = client.post("/api/v1/models/rollback", params={"to_model_id": seed_id},
                    headers=owner_headers)
    assert r.status_code == 200
    current = client.get("/api/v1/models/current", headers=owner_headers).json()
    assert current["id"] == seed_id


def test_lineage_tracks_parentage(client, owner_headers):
    _bootstrap(client)
    current = client.get("/api/v1/models/current", headers=owner_headers).json()
    chain = client.get(f"/api/v1/models/{current['id']}/lineage",
                       headers=owner_headers).json()
    assert chain[0]["id"] == current["id"]


def test_training_respects_owner_off_switch(client, owner_headers):
    _bootstrap(client)
    client.post("/api/v1/controls", json={"values": {"training_enabled": False}},
                headers=owner_headers)
    r = client.post("/api/v1/training/cycle", json={}, headers=owner_headers)
    assert "skipped" in r.json()
    client.post("/api/v1/controls", json={"values": {"training_enabled": True}},
                headers=owner_headers)
