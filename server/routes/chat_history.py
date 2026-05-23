"""Server-side chat history persistence.

Stores one JSON file per session in server/chat_history/. No auth, no
multi-tenancy — sessions are identified by a client-generated UUID.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from logger import logger

router = APIRouter()

HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "./chat_history")).resolve()
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _session_path(session_id: str) -> Path:
    if not _SESSION_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return HISTORY_DIR / f"{session_id}.json"


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    references: Optional[list] = None
    confidenceScore: Optional[int] = None
    status: Optional[str] = None


class SaveSessionRequest(BaseModel):
    session_id: str
    title: Optional[str] = None
    library: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)


@router.post("/chat/save/")
async def save_session(payload: SaveSessionRequest = Body(...)):
    path = _session_path(payload.session_id)
    body = {
        "session_id": payload.session_id,
        "title": payload.title or (payload.messages[0].content[:60] if payload.messages else "New chat"),
        "library": payload.library,
        "updatedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "messages": [m.model_dump() for m in payload.messages],
    }
    try:
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception(f"Saving session failed: {exc}")
        raise HTTPException(status_code=500, detail="Could not save session") from exc
    return {"ok": True, "session_id": payload.session_id, "updatedAt": body["updatedAt"]}


@router.get("/chat/list/")
async def list_sessions():
    sessions = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sessions.append(
            {
                "session_id": data.get("session_id") or f.stem,
                "title": data.get("title") or "(untitled)",
                "library": data.get("library"),
                "updatedAt": data.get("updatedAt"),
                "messageCount": len(data.get("messages") or []),
            }
        )
    sessions.sort(key=lambda s: s.get("updatedAt") or "", reverse=True)
    return {"sessions": sessions}


@router.get("/chat/{session_id}/")
async def get_session(session_id: str):
    path = _session_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read session") from exc


@router.delete("/chat/{session_id}/")
async def delete_session(session_id: str):
    path = _session_path(session_id)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Could not delete session") from exc
    return {"ok": True}
