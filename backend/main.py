import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import BaseModel

from rag import ingest_document, answer_question, UPLOAD_DIR

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

app = FastAPI(title="AI Document QA")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    destination = UPLOAD_DIR / filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        return ingest_document(destination)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (RuntimeError, OpenAIError) as e:
        raise HTTPException(500, str(e))


@app.post("/ask")
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    try:
        return answer_question(request.question, request.top_k)
    except (RuntimeError, OpenAIError) as e:
        raise HTTPException(500, str(e))
