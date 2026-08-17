from __future__ import annotations
import fnmatch, urllib.parse
from ..database import db

LEVELS = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
DEFAULT_APPROVAL_LEVELS = {"read": "LOW", "analyze": "LOW", "tests": "MEDIUM", "patch": "MEDIUM",
                           "code_edit": "HIGH", "dependency": "HIGH", "training": "HIGH",
                           "internet": "CRITICAL", "security_policy": "CRITICAL", "model_promote": "CRITICAL",
                           "owner_permissions": "CRITICAL"}

def kill_switch_active() -> bool: return bool(db.get_controls().get("kill_switch", False))
def allowed(capability: str) -> bool:
    c = db.get_controls()
    if kill_switch_active(): return False
    aliases = {"python":"python_enabled", "terminal":"terminal_enabled", "web":"web_enabled",
               "code_edit":"code_editing_enabled", "tools":"tools_enabled", "training":"training_enabled",
               "network":"internet_enabled", "external_models":"external_models_enabled", "files":"tools_enabled"}
    return bool(c.get(aliases.get(capability, capability), False))

def approval_level(action: str) -> str: return db.get_controls().get("approval_" + action, DEFAULT_APPROVAL_LEVELS.get(action, "HIGH"))

def domain_allowed(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    host = (p.hostname or "").lower()
    c = db.get_controls()
    if not c.get("internet_enabled", False) or not c.get("web_enabled", False): return False
    deny = c.get("domain_blocklist", []) or []
    allow = c.get("domain_allowlist", []) or []
    if any(fnmatch.fnmatch(host, x.lower()) for x in deny): return False
    return not allow or any(fnmatch.fnmatch(host, x.lower()) for x in allow)
