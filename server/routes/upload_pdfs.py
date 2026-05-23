from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
from modules.load_vectorstore import load_vectorstore, list_libraries
from fastapi.responses import JSONResponse
from logger import logger


router = APIRouter()


@router.post("/upload_pdfs/")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    library: Optional[str] = Form(None),
):
    try:
        logger.info(f"Received uploaded files for library={library!r}")
        namespace = load_vectorstore(files, library=library)
        logger.info(f"Documents added to library '{namespace}'")
        return {"message": "Files processed and vectorstore updated", "library": namespace}
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/libraries/")
async def get_libraries():
    return {"libraries": list_libraries()}
