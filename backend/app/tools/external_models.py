"""External-model gateway. Secrets come from environment only; providers never receive raw owner policy."""
from __future__ import annotations
import os, time, urllib.request, json
from ..database import db
from ..security.audit import log

class ExternalModelGateway:
    def call(self, provider: str, payload: dict, estimated_cost: float = 1.0):
        c=db.get_controls()
        if not c.get("external_models_enabled",False): raise PermissionError("external models disabled")
        from ..security.budget import check, charge
        check(estimated_cost)
        key=os.environ.get("OMNI_"+provider.upper()+"_API_KEY")
        if not key: raise RuntimeError("provider secret is not configured")
        # Provider adapters are explicit; no arbitrary URL is accepted from the model.
        urls={"openai":"https://api.openai.com/v1/chat/completions","anthropic":"https://api.anthropic.com/v1/messages"}
        if provider not in urls: raise ValueError("unsupported provider")
        req=urllib.request.Request(urls[provider],data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r: data=json.loads(r.read().decode())
        charge("external_api",estimated_cost,{"provider":provider}); log("agent","external_model_call",{"provider":provider,"ok":True})
        return data
external_gateway=ExternalModelGateway()
