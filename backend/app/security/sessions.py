from __future__ import annotations
import hashlib, secrets, time
from ..database import db

def _hash(token): return hashlib.sha256(token.encode()).hexdigest()
def create(device_id=None):
    token=secrets.token_urlsafe(48); now=time.time(); ttl=int(db.get_controls().get("session_ttl_s",86400))
    db.execute("INSERT INTO sessions(id,token_hash,role,device_id,created_at,expires_at) VALUES(?,?,?,?,?,?)",(db.new_id(),_hash(token),"owner",device_id,now,now+ttl))
    return token
def verify(token):
    if not token: return False
    row=db.query_one("SELECT * FROM sessions WHERE token_hash=?",(_hash(token),))
    return bool(row and not row["revoked"] and row["expires_at"]>time.time())
def revoke(token): db.execute("UPDATE sessions SET revoked=1 WHERE token_hash=?",(_hash(token),))
