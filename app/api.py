"""FastAPI backend.  Run: uvicorn app.api:app --host 0.0.0.0 --port 8000"""
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import config as C
from app.loaders import SUPPORTED
from app.rag import RagEngine

app = FastAPI(title="Document Q&A (RAG + Gemini)", version="1.0.0")
engine = RagEngine()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", **engine.stats()}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = Path(f.filename).name  # strip any path components
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file type: {name}. Use PDF, DOCX or TXT.")
        dest = C.UPLOAD_DIR / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    n = engine.add_files(saved)
    return {"files": [p.name for p in saved], "chunks_added": n, **engine.stats()}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty.")
    try:
        return engine.ask(req.question, req.session_id)
    except Exception as e:
        raise HTTPException(500, f"Generation failed (check GOOGLE_API_KEY): {e}")


@app.delete("/session/{session_id}")
def reset(session_id: str):
    engine.reset(session_id)
    return {"status": "reset"}
