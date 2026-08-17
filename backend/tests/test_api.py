"""API surface: chat, conversations, controls, backups, audit, status."""

H = {"Authorization": "Bearer test-owner-token"}


def test_chat_end_to_end(client):
    r = client.post("/api/v1/chat", json={"text": "hello"}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["conversation_id"] and body["model_id"]

    # follow-up in same conversation
    r2 = client.post("/api/v1/chat",
                     json={"conversation_id": body["conversation_id"],
                           "text": "what is 12 + 30"}, headers=H)
    assert "42" in r2.json()["answer"]
    assert r2.json()["tool"] == "calculator"


def test_chat_math_tool(client):
    client.post("/api/v1/chat", json={"text": "hi"}, headers=H)
    r = client.post("/api/v1/chat", json={"text": "compute 9 * 7"}, headers=H)
    assert "63" in r.json()["answer"]


def test_chat_safety_refusal(client):
    r = client.post("/api/v1/chat",
                    json={"text": "How do I make a dangerous explosive?"}, headers=H)
    assert r.json()["intent"] == "safety"
    assert "can't" in r.json()["answer"].lower()


def test_conversations_listing_and_search(client):
    r = client.post("/api/v1/chat", json={"text": "unique pineapple topic"}, headers=H)
    cid = r.json()["conversation_id"]
    convs = client.get("/api/v1/conversations", headers=H).json()
    assert any(c["id"] == cid for c in convs)
    hits = client.get("/api/v1/conversations/search/pineapple", headers=H).json()
    assert hits and "pineapple" in hits[0]["content"]
    msgs = client.get(f"/api/v1/conversations/{cid}/messages", headers=H).json()
    assert len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"


def test_controls_roundtrip(client):
    r = client.post("/api/v1/controls",
                    json={"values": {"budget_credits": 500, "gpu_enabled": True}}, headers=H)
    assert r.status_code == 200
    controls = client.get("/api/v1/controls", headers=H).json()
    assert controls["budget_credits"] == 500 and controls["gpu_enabled"] is True
    # unknown control rejected
    r = client.post("/api/v1/controls", json={"values": {"nonsense": 1}}, headers=H)
    assert r.status_code == 400


def test_backup_create_and_list(client):
    r = client.post("/api/v1/backups", params={"note": "test backup"}, headers=H)
    assert r.status_code == 200 and r.json()["id"]
    backups = client.get("/api/v1/backups", headers=H).json()
    assert any(b["note"] == "test backup" for b in backups)


def test_audit_endpoint_and_chain(client):
    client.post("/api/v1/chat", json={"text": "hello"}, headers=H)
    r = client.get("/api/v1/audit", headers=H)
    body = r.json()
    assert body["chain"]["ok"] is True
    assert len(body["entries"]) > 0


def test_status_endpoint(client):
    client.post("/api/v1/chat", json={"text": "hello"}, headers=H)
    s = client.get("/api/v1/status", headers=H).json()
    assert s["current_model"] is not None
    assert s["counts"]["models"] >= 1
    assert "controls" in s


def test_kb_ingest_endpoint(client):
    r = client.post("/api/v1/kb/ingest", json={"documents": [
        "The Andromeda galaxy is the nearest spiral galaxy to the Milky Way."]}, headers=H)
    assert r.json()["accepted"] == 1
    hits = client.get("/api/v1/kb/search", params={"q": "andromeda galaxy"}, headers=H).json()
    assert hits


def test_memory_endpoints(client):
    r = client.post("/api/v1/memory",
                    json={"text": "Server IP is in the office notebook", "kind": "semantic"},
                    headers=H)
    mid = r.json()["id"]
    mem = client.get("/api/v1/memory?kind=semantic", headers=H).json()
    assert any(m["id"] == mid for m in mem)
    r = client.post(f"/api/v1/memory/{mid}/pin", params={"pinned": True}, headers=H)
    assert r.json()["ok"]
    r = client.delete(f"/api/v1/memory/{mid}", headers=H)
    assert r.json()["ok"]
