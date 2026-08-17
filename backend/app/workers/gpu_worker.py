from __future__ import annotations
import queue, threading, uuid, time
from ..database import db
from ..security import audit

class GPUWorker:
    def __init__(self,max_concurrent=1):
        self.q=queue.Queue(); self.jobs={}; self.stop=threading.Event(); self.max_concurrent=max_concurrent; self.threads=[]
    def detect(self):
        try:
            import torch
            return {"available":torch.cuda.is_available(),"count":torch.cuda.device_count(),"vram_mb":int(torch.cuda.get_device_properties(0).total_memory/1048576) if torch.cuda.is_available() else 0}
        except Exception as e:return {"available":False,"count":0,"vram_mb":0,"error":str(e)}
    def start(self):
        if self.threads:return
        for _ in range(self.max_concurrent):
            t=threading.Thread(target=self._loop,daemon=True); t.start(); self.threads.append(t)
    def submit(self,kind,payload):
        if db.get_controls().get("kill_switch"): raise PermissionError("kill switch active")
        jid=uuid.uuid4().hex[:16]; self.jobs[jid]={"id":jid,"kind":kind,"status":"queued","created_at":time.time()}; self.q.put((jid,kind,payload)); return self.jobs[jid]
    def cancel(self,jid):
        if jid in self.jobs:self.jobs[jid]["cancelled"]=True; return True
        return False
    def _loop(self):
        while not self.stop.is_set():
            try: jid,kind,payload=self.q.get(timeout=.2)
            except queue.Empty: continue
            job=self.jobs[jid]; job["status"]="running"
            try:
                if job.get("cancelled") or db.get_controls().get("kill_switch"): raise RuntimeError("cancelled")
                if kind=="inference":
                    from ..llm.engine import llm_engine,GenerationConfig
                    if not llm_engine.loaded: raise RuntimeError("no model loaded")
                    job["result"]=llm_engine.generate(payload["messages"],GenerationConfig(**payload.get("config",{})))
                elif kind=="training":
                    from ..training.trainer import trainer
                    job["result"]=trainer.train_candidate(**payload)
                else: raise ValueError("unknown job kind")
                job["status"]="completed"
            except Exception as e: job["status"]="failed"; job["error"]=str(e)
            job["finished_at"]=time.time(); audit.log("gpu_worker","job_finished",job); self.q.task_done()
    def shutdown(self): self.stop.set()

gpu_worker=GPUWorker()
