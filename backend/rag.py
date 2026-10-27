import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from docx import Document
from openai import OpenAI
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR.mkdir(exist_ok=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
    "documents_local",
    embedding_function=DefaultEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)
_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to backend/.env and restart the server.")
        _client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    return _client


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        return "\n".join(p.text for p in Document(str(file_path)).paragraphs)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def ingest_document(file_path) -> dict:
    file_path = Path(file_path)
    chunks = chunk_text(extract_text(file_path))
    if not chunks:
        raise ValueError(f"No extractable text found in {file_path.name}")

    source = file_path.name
    _collection.delete(where={"source": source})
    _collection.add(
        ids=[f"{source}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
    )
    return {"filename": source, "chunks": len(chunks)}


def search_documents(query: str, top_k: int = 4, source: Optional[str] = None) -> List[dict]:
    count = _collection.count()
    if count == 0:
        return []
    results = _collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        where={"source": source} if source else None,
    )
    return [
        {"source": meta["source"], "chunk": meta["chunk"], "text": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def list_documents() -> List[str]:
    metadatas = _collection.get(include=["metadatas"])["metadatas"]
    return sorted({meta["source"] for meta in metadatas})
