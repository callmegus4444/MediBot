import json
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse

from logger import logger
from modules.strict_orchestrator import answer_strict, stream_answer_strict
from schemas.strict import abstention_response

router = APIRouter()


def _parse_history(history_json: Optional[str]):
    if not history_json:
        return []
    try:
        parsed = json.loads(history_json)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        logger.warning("Bad history_json — ignoring")
    return []


@router.post("/ask/strict/")
async def ask_question_strict(
    question: str = Form(...),
    library: Optional[str] = Form(None),
    use_web: bool = Form(True),
    history_json: Optional[str] = Form(None),
):
    logger.info(f"strict query: lib={library!r} use_web={use_web} q={question[:80]!r}")
    try:
        response = answer_strict(
            question,
            library=library,
            use_web=use_web,
            history=_parse_history(history_json),
        )
        return JSONResponse(content=response.model_dump())
    except Exception as exc:
        logger.exception(f"Strict orchestration failed: {exc}")
        fallback = abstention_response(request_id="error")
        return JSONResponse(status_code=500, content=fallback.model_dump())


@router.post("/ask/strict/stream/")
async def ask_question_strict_stream(
    question: str = Form(...),
    library: Optional[str] = Form(None),
    use_web: bool = Form(True),
    history_json: Optional[str] = Form(None),
):
    logger.info(f"strict-stream query: lib={library!r} use_web={use_web} q={question[:80]!r}")
    history = _parse_history(history_json)

    def event_gen():
        try:
            for event in stream_answer_strict(
                question, library=library, use_web=use_web, history=history
            ):
                payload = json.dumps(event.get("data"), ensure_ascii=False)
                yield f"event: {event.get('event','message')}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.exception(f"Strict stream failed: {exc}")
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
