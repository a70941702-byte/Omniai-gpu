"""Memory layers, knowledge base ingestion pipeline, owner data control."""


def test_memory_layers_and_recall():
    from app.memory.store import memory_store
    memory_store.remember("My favorite color is blue", kind="semantic", importance=0.9)
    memory_store.remember("We discussed physics yesterday", kind="episodic")
    hits = memory_store.recall("favorite color", kinds=("semantic",))
    assert hits and "blue" in hits[0]["text"]


def test_memory_respects_learning_toggle():
    from app.database import db
    from app.memory.store import memory_store
    db.set_control("learning_enabled", False)
    mid = memory_store.remember("should not be stored", kind="semantic")
    assert mid is None
    db.set_control("learning_enabled", True)


def test_kb_rejects_pii_and_duplicates():
    from app.knowledge.base import knowledge_base
    doc_ok = "The mitochondria is the powerhouse of the cell and produces ATP energy."
    doc_pii = "Call me at 555-123-4567 for the secret documents and more info."
    r1 = knowledge_base.ingest([doc_ok, doc_pii, doc_ok])  # 3rd is a duplicate
    assert r1["accepted"] == 1
    assert r1["rejected_privacy"] == 1
    assert r1["rejected_duplicate"] == 1


def test_kb_retrieval_after_ingest():
    from app.knowledge.base import knowledge_base
    knowledge_base.ingest(["Python decorators wrap functions to extend their behavior cleanly."])
    hits = knowledge_base.retrieve("python decorators wrap functions")
    assert hits and hits[0]["score"] > 0.2


def test_owner_can_forget_memory():
    from app.memory.store import memory_store
    mid = memory_store.remember("temporary fact", kind="semantic")
    memory_store.forget(mid, actor="owner")
    from app.database import db
    assert db.query_one("SELECT id FROM memory WHERE id=?", (mid,)) is None
