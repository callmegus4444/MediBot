from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from logger import logger
from modules.strict_orchestrator import answer_strict
from schemas.strict import abstention_response

router = APIRouter()


@router.post("/ask/strict/")
async def ask_question_strict(question: str = Form(...)):
    logger.info(f"strict-mode user query: {question}")
    try:
        response = answer_strict(question)
        return JSONResponse(content=response.model_dump())
    except Exception as exc:
        logger.exception(f"Strict orchestration failed: {exc}")
        fallback = abstention_response(request_id="error")
        return JSONResponse(status_code=500, content=fallback.model_dump())
