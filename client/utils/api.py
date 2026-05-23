import json
from typing import Iterable, Optional

import requests
from config import API_URL


def upload_pdfs_api(files, library: Optional[str] = None):
    files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in files]
    data = {"library": library} if library else None
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload, data=data)


def list_libraries_api():
    try:
        r = requests.get(f"{API_URL}/libraries/", timeout=10)
        if r.status_code == 200:
            return r.json().get("libraries", []) or []
    except requests.RequestException:
        pass
    return []


def ask_question(question):
    return requests.post(f"{API_URL}/ask/", data={"question": question})


def _strict_payload(question, library, use_web, history):
    return {
        "question": question,
        "library": library or "",
        "use_web": "true" if use_web else "false",
        "history_json": json.dumps(history or []),
    }


def ask_question_strict(question, library=None, use_web=True, history=None):
    return requests.post(
        f"{API_URL}/ask/strict/",
        data=_strict_payload(question, library, use_web, history),
        timeout=90,
    )


def stream_strict(question, library=None, use_web=True, history=None) -> Iterable[dict]:
    """Yield SSE events as dicts: {"event": str, "data": Any}."""
    with requests.post(
        f"{API_URL}/ask/strict/stream/",
        data=_strict_payload(question, library, use_web, history),
        stream=True,
        timeout=180,
    ) as resp:
        if resp.status_code != 200:
            yield {"event": "error", "data": {"message": resp.text}}
            return
        event_name = "message"
        data_buf = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r")
            if line == "":
                if data_buf:
                    raw = "\n".join(data_buf)
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        data = raw
                    yield {"event": event_name, "data": data}
                event_name = "message"
                data_buf = []
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_buf.append(line[len("data:"):].lstrip())


def list_sessions_api():
    try:
        r = requests.get(f"{API_URL}/chat/list/", timeout=10)
        if r.status_code == 200:
            return r.json().get("sessions", []) or []
    except requests.RequestException:
        pass
    return []


def get_session_api(session_id: str):
    try:
        r = requests.get(f"{API_URL}/chat/{session_id}/", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def save_session_api(session_id: str, messages, title=None, library=None):
    try:
        return requests.post(
            f"{API_URL}/chat/save/",
            json={
                "session_id": session_id,
                "title": title,
                "library": library,
                "messages": messages,
            },
            timeout=15,
        )
    except requests.RequestException as exc:
        return None


def delete_session_api(session_id: str):
    try:
        return requests.delete(f"{API_URL}/chat/{session_id}/", timeout=10)
    except requests.RequestException:
        return None
