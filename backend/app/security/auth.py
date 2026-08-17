"""Owner authentication and the immutable privilege model.

Privilege rules (enforced here, not by convention):
- The owner authenticates with a bearer token.
- The AI itself NEVER receives owner scope. Internal subsystems act as the
  'system' actor and can only call owner-gated functions indirectly through
  the Approval System (which records a pending request; execution happens
  only after the owner approves).
- There is no API to grant new privileges, change the owner token, disable
  the approval system, or delete audit entries. Those operations simply do
  not exist in the codebase.
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..config import load_or_create_owner_token
from .sessions import verify as verify_session

_OWNER_TOKEN = load_or_create_owner_token()


def owner_token() -> str:
    return _OWNER_TOKEN


def verify_owner_token(token: str) -> bool:
    return hmac.compare_digest(token, _OWNER_TOKEN)


async def require_owner(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not (verify_owner_token(token) or verify_session(token)):
        raise HTTPException(status_code=403, detail="invalid or expired owner session")
    return "owner"
