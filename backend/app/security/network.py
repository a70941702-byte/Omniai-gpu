from __future__ import annotations
import urllib.request
from .policy import domain_allowed

def get(url: str, timeout=10) -> str:
    if not domain_allowed(url): raise PermissionError("network request denied by owner policy")
    req=urllib.request.Request(url,headers={"User-Agent":"OmniAI/1.0"})
    with urllib.request.urlopen(req, timeout=min(timeout,30)) as r: return r.read(2*1024*1024).decode("utf-8","replace")
