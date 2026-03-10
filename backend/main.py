"""
FinSight - RAG-Powered Financial Intelligence Assistant
FastAPI Backend - runs locally on http://localhost:8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys

sys.path.append(os.path.dirname(__file__))
from rag_pipeline import FinSightRAG

app = FastAPI(title="FinSight API", version="1.0.0", description="RAG-Powered Financial Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing FinSight RAG pipeline...")
rag = FinSightRAG()
print("RAG pipeline ready.")


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    use_rag: bool = True
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: list
    use_rag: bool
    grounding_score: Optional[float] = None
    chunks_retrieved: int = 0

class IngestTextRequest(BaseModel):
    text: str
    source_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "FinSight API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "vector_db": rag.get_collection_stats()}

@app.get("/stats")
def get_stats():
    return rag.get_collection_stats()

@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    chunks = rag.ingest_text(request.text, request.source_name)
    return {"message": "Ingested successfully", "chunks_created": chunks, "document_name": request.source_name}

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")
    chunks = rag.ingest_pdf(contents, file.filename)
    return {"message": "PDF ingested successfully", "chunks_created": chunks, "document_name": file.filename}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the assistant.
    use_rag=True  -> grounded response with citations
    use_rag=False -> raw LLM response (hallucination risk)
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if request.use_rag:
        result = rag.answer_with_rag(request.question, top_k=request.top_k)
    else:
        result = rag.answer_without_rag(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        use_rag=request.use_rag,
        grounding_score=result.get("grounding_score"),
        chunks_retrieved=result.get("chunks_retrieved", 0)
    )

@app.post("/compare")
def compare(request: QueryRequest):
    """Side-by-side RAG vs non-RAG — the hallucination demo."""
    rag_result = rag.answer_with_rag(request.question, top_k=request.top_k)
    non_rag_result = rag.answer_without_rag(request.question)
    return {
        "question": request.question,
        "rag_response": {
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "grounding_score": rag_result.get("grounding_score"),
            "chunks_retrieved": rag_result.get("chunks_retrieved", 0)
        },
        "non_rag_response": {
            "answer": non_rag_result["answer"],
            "sources": [],
            "grounding_score": None,
            "note": "Training data only. May hallucinate specific financial figures."
        }
    }

@app.delete("/reset")
def reset_collection():
    rag.chroma_client.delete_collection("finsight_docs")
    rag.collection = rag.chroma_client.get_or_create_collection(
        name="finsight_docs",
        embedding_function=rag.embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return {"message": "Vector database cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
