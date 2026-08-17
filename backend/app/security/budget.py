from __future__ import annotations
from ..database import db
from ..security import audit

def check(cost: float) -> None:
    c=db.get_controls(); remaining=float(c.get("budget_credits",0))-float(c.get("budget_spent",0))
    if cost>remaining: raise PermissionError("budget exceeded")
def charge(category: str, amount: float, metadata=None):
    check(amount); db.set_control("budget_spent", float(db.get_controls().get("budget_spent",0))+float(amount))
    db.execute("INSERT INTO costs(id,category,amount,metadata,created_at) VALUES(?,?,?,?,?)",(db.new_id(),category,float(amount),__import__("json").dumps(metadata or {}),db.now()))
    audit.log("system","cost_recorded",{"category":category,"credits":amount,"metadata":metadata or {}})
