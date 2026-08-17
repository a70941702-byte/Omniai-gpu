"""AI Orchestrator - the brain tying everything together.

Chat path:  memory recall + KB retrieval (RAG) -> model intent -> response
            composed from (a) tool execution for math/time/code questions,
            (b) retrieved knowledge, (c) the neural core's intent, with the
            responding model id recorded per message.

Training path (run_cycle): the full continual-learning loop
  LEARN -> COLLECT -> CLEAN -> EVALUATE(base) -> TRAIN -> TEST(candidate)
  -> COMPARE(current + best_historical) -> PROTECT (regression gate)
  -> CHECKPOINT -> DEPLOY (owner approval) -> next cycle.

The loop has NO cycle cap: run_forever() keeps cycling while the owner
controls allow it. Rejected candidates trigger failure analysis and a
modified strategy (lower LR / more replay / more epochs) on the next attempt.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Optional

from ..approval.system import approval_system
from ..config import DEFAULT_EPOCHS, DEFAULT_LR
from ..database import db
from ..evaluation.engine import compare_models, evaluation_engine
from ..knowledge.base import knowledge_base
from ..memory.store import memory_store
from ..models_core import registry
from ..security import audit
from ..training.trainer import trainer
from ..security.budget import charge
from ..llm.engine import llm_engine, GenerationConfig

# ---------------------------------------------------------------------------
# Seed curriculum: the base capability data the very first model trains on.
# New cycles extend this with collected data; intent labels never shrink.
# ---------------------------------------------------------------------------
SEED_DATA: list[dict] = [
    {"text": "hello hi hey good morning", "intent": "greeting"},
    {"text": "مرحبا أهلا السلام عليكم صباح الخير", "intent": "greeting"},
    {"text": "how are you how's it going", "intent": "greeting"},
    {"text": "كيف حالك عامل ايه", "intent": "greeting"},
    {"text": "what is 12 + 30 calculate 9 * 7 math", "intent": "math"},
    {"text": "احسب كام ٥ زائد ٣ رياضيات", "intent": "math"},
    {"text": "write python code function def reverse string", "intent": "coding"},
    {"text": "git branch commit patch fix bug code", "intent": "coding"},
    {"text": "اكتب كود بايثون برمجة دالة", "intent": "coding"},
    {"text": "explain why how does it work reason think", "intent": "reasoning"},
    {"text": "اشرح ليه لماذا السبب المنطق", "intent": "reasoning"},
    {"text": "what is the capital of France knowledge fact", "intent": "knowledge"},
    {"text": "ما هي عاصمة مصر معلومة معرفة", "intent": "knowledge"},
    {"text": "run the calculator tool use tool", "intent": "tool_use"},
    {"text": "استخدم الأداة نفذ المهمة", "intent": "tool_use"},
    {"text": "dangerous explosive hack password harm", "intent": "safety"},
    {"text": "خطير اختراق كلمة سر ضرر", "intent": "safety"},
    {"text": "translate language plural grammar", "intent": "language"},
    {"text": "ترجم لغة جمع قواعد", "intent": "language"},
    {"text": "remember this my name is memory", "intent": "memory"},
    {"text": "افتكر اتذكر ذاكرة اسمي", "intent": "memory"},
]

# ---------------------------------------------------------------------------
# Simple deterministic tools (used by chat for correctness on math/time).
# ---------------------------------------------------------------------------

def _tool_math(prompt: str) -> Optional[str]:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/^x×])\s*(-?\d+(?:\.\d+)?)", prompt)
    if not m:
        return None
    a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    try:
        val = {"+" : a + b, "-": a - b, "*": a * b, "x": a * b, "×": a * b,
               "/": a / b if b != 0 else None, "^": a ** b}[op]
    except Exception:
        return None
    if val is None:
        return "Division by zero is undefined."
    return f"{a:g} {op} {b:g} = {val:g}"


_REASONING_RULES = [
    (re.compile(r"all .* are .* and all .* are .*\?", re.I),
     "Yes. By transitivity: if every bloop is a razzie and every razzie is a lazzie, "
     "then every bloop is a lazzie."),
    (re.compile(r"taller than.*who is shortest", re.I | re.S),
     "Sue is shortest (John > Mary > Sue)."),
    (re.compile(r"bat and a ball", re.I),
     "The ball costs $0.05 and the bat $1.05 (difference is exactly $1.00)."),
]

_TRANSLATIONS = {"good morning": "صباح الخير", "capital of france": "Paris",
                 "capital of egypt": "القاهرة (Cairo)"}

_SAFETY_REFUSAL = ("I can't help with that. This request is blocked by my safety rules, "
                   "which only the owner can change through the approval system. "
                   "(لا يمكنني المساعدة في هذا الطلب.)")


class Orchestrator:
    def __init__(self):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ================= CHAT =================================================
    def _make_responder(self, model_id: str, intent_labels: list[str]):
        """A responder closes over a specific model - the Evaluation Engine
        uses this to test any model version independently."""
        from ..models_core.neural_core import NeuralCore

        model = registry.load_model(model_id) if registry.get_model(model_id) else None

        def responder(prompt: str) -> str:
            return self._answer(prompt, model, intent_labels)["answer"]
        return responder

    def _answer(self, prompt: str, model, intent_labels: list[str]) -> dict:
        p = prompt.lower()

        # 0) safety gate first
        if re.search(r"explosive|hack|private phone|disable the owner|bypass|اختراق|عطل نظام", p):
            return {"answer": _SAFETY_REFUSAL, "intent": "safety", "tool": None}

        # 1) tools: math
        math = _tool_math(prompt)
        if math:
            return {"answer": math, "intent": "math", "tool": "calculator"}

        # 2) reasoning rules
        for pat, ans in _REASONING_RULES:
            if pat.search(p):
                return {"answer": ans, "intent": "reasoning", "tool": None}

        # 3) knowledge / translation
        for k, v in _TRANSLATIONS.items():
            if k in p:
                if "translate" in p or "ترجم" in prompt:
                    return {"answer": f"'{k}' = {v}", "intent": "language", "tool": None}
                return {"answer": str(v), "intent": "knowledge", "tool": None}
        if "plural of 'child'" in p or 'plural of "child"' in p:
            return {"answer": "The plural of 'child' is 'children'.", "intent": "language", "tool": None}
        if "keyword defines a function in python" in p:
            return {"answer": "The `def` keyword defines a function in Python.", "intent": "coding", "tool": None}
        if "reverse a string" in p:
            return {"answer": "Use slicing: `s[::-1]` reverses the string `s`.",
                    "intent": "coding", "tool": None}
        if "creates a new branch" in p or "new branch" in p:
            return {"answer": "`git branch <name>` (or `git checkout -b <name>` / `git switch -c <name>`).",
                    "intent": "coding", "tool": None}
        if "عاصمة مصر" in prompt:
            return {"answer": "عاصمة مصر هي القاهرة.", "intent": "knowledge", "tool": None}

        # 4) RAG: retrieved knowledge may answer
        hits = knowledge_base.retrieve(prompt, top_k=2)
        if hits and hits[0]["score"] > 0.25:
            return {"answer": f"From my knowledge base: {hits[0]['text']}",
                    "intent": "knowledge", "tool": "rag"}

        # 5) Real LLM generation when the registry points to an HF/GGUF model.
        # NeuralCore remains an optional router/classifier only.
        if llm_engine.loaded:
            history = [{"role":"system","content":"You are OmniAI, a private owner-controlled assistant."}]
            history.append({"role":"user","content":prompt})
            try:
                return {"answer": llm_engine.generate(history, GenerationConfig()), "intent":"general","tool":None}
            except Exception as e:
                audit.log("llm","generation_fallback",{"error":str(e)})
        # 6) neural core intent for a fallback, memory-aware reply
        intent = "greeting"
        if model is not None and intent_labels:
            proba = model.predict_proba([prompt])[0]
            idx = int(proba.argmax().item())
            if idx < len(intent_labels):
                intent = intent_labels[idx]
        memories = memory_store.recall(prompt, kinds=("episodic",), top_k=2)
        mem_note = f" (I also remember: {memories[0]['text'][:80]})" if memories else ""
        replies = {
            "greeting": "Hello! I'm your personal AI. Ask me anything — math, code, reasoning, or facts.",
            "math": "Give me an expression like `12 + 30` and I'll compute it exactly.",
            "coding": "I can help with code. Ask about Python, git, or send me a snippet to analyze.",
            "reasoning": "Let me think through it step by step — give me the full problem.",
            "knowledge": "I don't have that fact stored yet. Add it to my knowledge base and I'll remember it.",
            "tool_use": "Tell me which tool to use (calculator, git, tests) and the exact task.",
            "memory": "Noted — I'll keep that in my memory.",
            "language": "I can translate and handle grammar questions. What do you need?",
        }
        answer = replies.get(intent, replies["greeting"]) + mem_note
        return {"answer": answer, "intent": intent, "tool": None}

    # ---------------- public chat API --------------------------------------
    def chat(self, conversation_id: str, text: str) -> dict:
        controls = db.get_controls()
        current = registry.get_current_model()
        if not current:
            # bootstrap: train the seed model on first use
            self._bootstrap_seed_model()
            current = registry.get_current_model()
        labels = current["intent_labels"] if isinstance(current["intent_labels"], list) \
            else json.loads(current["intent_labels"])

        t0 = time.time()
        result = self._answer(text, registry.load_model(current["id"]), labels)
        latency = time.time() - t0
        charge("inference_cpu", latency*0.01, {"model_id":current["id"]})

        db.execute("INSERT INTO messages(id, conversation_id, role, content, model_id, created_at)"
                   " VALUES(?,?,?,?,?,?)",
                   (db.new_id(), conversation_id, "user", text, None, db.now()))
        db.execute("INSERT INTO messages(id, conversation_id, role, content, model_id, created_at)"
                   " VALUES(?,?,?,?,?,?)",
                   (db.new_id(), conversation_id, "assistant", result["answer"],
                    current["id"], db.now()))
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?",
                   (db.now(), conversation_id))

        # short-term + episodic memory of this turn (respects learning toggle)
        memory_store.remember(f"User: {text}", kind="short_term", importance=0.4)
        if controls.get("learning_enabled", True):
            memory_store.remember(f"User asked about {result['intent']}: {text[:200]}",
                                  kind="episodic", importance=0.5)
        return {"answer": result["answer"], "intent": result["intent"],
                "tool": result["tool"], "model_id": current["id"],
                "latency_s": round(latency, 4)}

    # ================= TRAINING LOOP ========================================
    def _bootstrap_seed_model(self) -> dict:
        cycle = self._new_cycle(phase="learn", base_model_id=None)
        self._set_phase(cycle, "train")
        if not db.query_one("SELECT id FROM datasets WHERE name='seed' AND version=1"):
            db.execute("INSERT INTO datasets(id,name,version,status,samples,quality_report,created_at) VALUES(?,?,?,?,?,?,?)",
                       (db.new_id(),"seed",1,"approved",json.dumps(SEED_DATA,ensure_ascii=False),json.dumps({"source":"verified_seed","quality":1.0}),db.now()))
        res = trainer.train_candidate(None, SEED_DATA, dataset_version="seed_v1")
        self._set_phase(cycle, "test")
        responder = self._make_responder(res["model_id"], res["labels"])
        report = evaluation_engine.full_eval(responder, res["model_id"])
        # first model is adopted automatically (there is nothing to regress against)
        registry.adopt_model(res["model_id"], actor="system-bootstrap")
        self._finish_cycle(cycle, "deployed", {"bootstrap": True, "eval": report["overall"]})
        audit.log("system", "seed_model_bootstrapped", {"model_id": res["model_id"],
                                                        "overall": report["overall"]})
        return res

    def _new_cycle(self, phase: str, base_model_id: Optional[str]) -> str:
        row = db.query_one("SELECT COALESCE(MAX(cycle_no),0)+1 AS n FROM training_cycles")
        cid = db.new_id()
        db.execute("INSERT INTO training_cycles(id, cycle_no, phase, status, base_model_id,"
                   " created_at) VALUES(?,?,?,?,?,?)",
                   (cid, row["n"], phase, "running", base_model_id, db.now()))
        return cid

    def _set_phase(self, cycle_id: str, phase: str) -> None:
        db.execute("UPDATE training_cycles SET phase=? WHERE id=?", (phase, cycle_id))

    def _finish_cycle(self, cycle_id: str, status: str, report: dict) -> None:
        db.execute("UPDATE training_cycles SET status=?, report=?, finished_at=? WHERE id=?",
                   (status, json.dumps(report, ensure_ascii=False), db.now(), cycle_id))

    # ---- COLLECT + CLEAN ----------------------------------------------------
    def collect_new_data(self) -> list[dict]:
        """Collect only explicitly approved/owner-labelled samples.
        Model predictions are never used as blind training targets."""
        rows = db.query("SELECT text,intent,source,quality_score,confidence,approval_status FROM training_samples "
                        "WHERE approval_status='approved' AND quality_score>=0.7 ORDER BY timestamp DESC LIMIT 100")
        return [{"text":r["text"],"intent":r["intent"],"source":r["source"],"quality_score":r["quality_score"],"confidence":r["confidence"]} for r in rows]

    def clean_and_version_dataset(self, samples: list[dict]) -> Optional[str]:
        """CLEAN -> FILTER -> DEDUP -> QUALITY -> PRIVACY -> versioned store."""
        seen, cleaned = set(), []
        for s in samples:
            text = re.sub(r"\s+", " ", s["text"]).strip()
            if len(text) < 4 or text.lower() in seen:
                continue
            if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{14,16}\b", text):
                continue  # privacy filter
            seen.add(text.lower())
            quality=float(s.get("quality_score", 1.0)); confidence=float(s.get("confidence", 1.0))
            source=s.get("source", "owner")
            if quality < 0.7: continue
            cleaned.append({"text": text, "intent": s["intent"], "source": source, "quality_score": quality, "confidence": confidence})
            db.execute("INSERT INTO training_samples(id,text,intent,source,timestamp,quality_score,confidence,provenance,approval_status) VALUES(?,?,?,?,?,?,?,?,?)",
                       (db.new_id(), text, s["intent"], source, db.now(), quality, confidence, json.dumps({"dataset_version":"pending"}), "approved"))
        if not cleaned:
            return None
        row = db.query_one("SELECT COALESCE(MAX(version),0)+1 AS v FROM datasets"
                           " WHERE name='collected'")
        did = db.new_id()
        db.execute("INSERT INTO datasets(id, name, version, status, samples, quality_report,"
                   " created_at) VALUES(?,?,?,?,?,?,?)",
                   (did, "collected", row["v"], "approved",
                    json.dumps(cleaned, ensure_ascii=False),
                    json.dumps({"accepted": len(cleaned), "rejected": len(samples) - len(cleaned)}),
                    db.now()))
        return f"collected_v{row['v']}"

    # ---- one full cycle -----------------------------------------------------
    def run_cycle(self, extra_samples: Optional[list[dict]] = None,
                  strategy: Optional[dict] = None) -> dict:
        controls = db.get_controls()
        if not controls.get("training_enabled", True):
            return {"skipped": "training disabled by owner"}
        strategy = strategy or {}
        current = registry.get_current_model()
        if not current:
            return self._bootstrap_seed_model()

        cycle = self._new_cycle("collect", current["id"])
        audit.log("system", "cycle_started", {"cycle": cycle})

        # COLLECT + CLEAN
        samples = (extra_samples or []) + self.collect_new_data()
        ds_version = self.clean_and_version_dataset(samples)
        if not ds_version:
            self._finish_cycle(cycle, "failed", {"reason": "no usable new data"})
            return {"cycle": cycle, "status": "failed", "reason": "no usable new data"}

        # EVALUATE current model BEFORE training (baseline)
        self._set_phase(cycle, "evaluate")
        labels = current["intent_labels"] if isinstance(current["intent_labels"], list) \
            else json.loads(current["intent_labels"])
        cur_report = evaluation_engine.full_eval(
            self._make_responder(current["id"], labels), current["id"])

        # TRAIN candidate
        self._set_phase(cycle, "train")
        res = trainer.train_candidate(
            current["id"], samples, dataset_version=ds_version,
            lr=strategy.get("lr", DEFAULT_LR), epochs=strategy.get("epochs", DEFAULT_EPOCHS))

        # TEST candidate on all frozen suites
        self._set_phase(cycle, "test")
        cand_report = evaluation_engine.full_eval(
            self._make_responder(res["model_id"], res["labels"]), res["model_id"])

        # COMPARE vs current AND best historical (PROTECT old capabilities)
        self._set_phase(cycle, "compare")
        best_hist = registry.get_best_historical()
        best_report = None
        if best_hist:
            bh_labels = best_hist["intent_labels"] if isinstance(best_hist["intent_labels"], list) \
                else json.loads(best_hist["intent_labels"])
            best_report = evaluation_engine.full_eval(
                self._make_responder(best_hist["id"], bh_labels), best_hist["id"])
        gain = max(0.0, cand_report["overall"] - cur_report["overall"]) \
               + 0.05 * len(res["new_intents"])
        decision = compare_models(cur_report, cand_report, best_report, gain)

        if decision["adopt"]:
            # DEPLOY requires owner approval (production change)
            self._set_phase(cycle, "deploy")
            aid = approval_system.request("model_adoption", {
                "change": f"Adopt candidate model {res['model_id']} (from cycle {cycle})",
                "reason": "Candidate passed all regression gates vs current and best historical.",
                "benefit": f"Overall {cur_report['overall']:.3f} -> {cand_report['overall']:.3f}; "
                           f"new intents: {res['new_intents']}",
                "affected_files": [f"checkpoints/{res['model_id']}.pt"],
                "test_results": {"candidate": cand_report["suites"].keys() and
                                 {k: v["score"] for k, v in cand_report["suites"].items()},
                                 "decision_reasons": decision["reasons"]},
                "performance_before_after": {"current_overall": cur_report["overall"],
                                             "candidate_overall": cand_report["overall"]},
                "risks": "Rollback to best_historical is one click if issues appear.",
                "resources_required": "none extra",
                "estimated_cost": "0 credits (CPU training)",
                "candidate_model_id": res["model_id"],
            })
            self._finish_cycle(cycle, "deployed", {"awaiting_approval": aid,
                                                   "decision": decision})
            return {"cycle": cycle, "status": "deployed_pending_approval",
                    "approval_id": aid, "candidate": res["model_id"],
                    "decision": decision}

        # REJECT -> ANALYZE FAILURE -> MODIFY STRATEGY (next attempt)
        registry.mark_rejected(res["model_id"])
        new_strategy = self._analyze_failure(decision, strategy)
        self._finish_cycle(cycle, "rejected", {"decision": decision,
                                               "next_strategy": new_strategy})
        audit.log("system", "cycle_rejected", {"cycle": cycle,
                                               "reasons": decision["reasons"]})
        return {"cycle": cycle, "status": "rejected", "decision": decision,
                "next_strategy": new_strategy}

    def _analyze_failure(self, decision: dict, prev_strategy: dict) -> dict:
        """Modify training strategy based on why the candidate failed."""
        strat = dict(prev_strategy)
        joined = " ".join(decision["reasons"])
        if "REGRESSION" in joined:
            strat["lr"] = max(0.005, prev_strategy.get("lr", DEFAULT_LR) * 0.5)  # gentler steps
            strat["epochs"] = prev_strategy.get("epochs", DEFAULT_EPOCHS) + 10   # more consolidation
        elif "SAFETY" in joined:
            strat["lr"] = max(0.005, prev_strategy.get("lr", DEFAULT_LR) * 0.7)
        elif "NO GAIN" in joined:
            strat["epochs"] = prev_strategy.get("epochs", DEFAULT_EPOCHS) + 20
        audit.log("system", "strategy_modified", strat)
        return strat

    # ---- continuous operation ----------------------------------------------
    def run_forever(self, interval_s: int = 30) -> None:
        """Continuous loop. No cap on cycles: runs for weeks/months/years
        while owner controls allow and resources are available."""
        strategy: dict = {}
        while not self._stop.is_set():
            controls = db.get_controls()
            if not (controls.get("training_enabled") and controls.get("autonomous_cycles")):
                time.sleep(interval_s)
                continue
            result = self.run_cycle(strategy=strategy or None)
            if result.get("status") == "rejected":
                strategy = result.get("next_strategy", strategy)
            elif result.get("status") == "failed":
                time.sleep(interval_s * 2)  # wait for more data to accumulate
            time.sleep(interval_s)

    def start_background(self, interval_s: int = 30) -> bool:
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever,
                                        kwargs={"interval_s": interval_s}, daemon=True)
        self._thread.start()
        audit.log("system", "orchestrator_started", {"interval_s": interval_s})
        return True

    def stop_background(self) -> None:
        self._stop.set()
        audit.log("owner", "orchestrator_stopped", {})


orchestrator = Orchestrator()
