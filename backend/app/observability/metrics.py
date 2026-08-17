from __future__ import annotations
import time, threading, logging
log=logging.getLogger("omniai")
_lock=threading.Lock(); counters={}; latencies=[]
def record(name, value=1):
    with _lock: counters[name]=counters.get(name,0)+value
def snapshot():
    with _lock: return {"counters":dict(counters),"latency_count":len(latencies),"latency_avg_s":sum(latencies)/len(latencies) if latencies else 0}
class MetricsMiddleware:
    def __init__(self,app): self.app=app
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http": return await self.app(scope,receive,send)
        t=time.monotonic(); record("http_requests_total")
        async def wrapped_send(message):
            if message["type"]=="http.response.start": record(f"http_status_{message['status']}")
            await send(message)
        try: return await self.app(scope,receive,wrapped_send)
        finally:
            elapsed=time.monotonic()-t
            with _lock: latencies.append(elapsed)
            log.info("request",extra={"path":scope.get("path"),"method":scope.get("method"),"latency_s":elapsed})
