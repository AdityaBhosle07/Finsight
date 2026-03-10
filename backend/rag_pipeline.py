"""
FinSight RAG Pipeline
Handles: document chunking, embeddings, vector storage, retrieval, LLM grounding
"""

import os
import re
import hashlib
from typing import Optional
import ollama
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import io


class FinSightRAG:
    def __init__(self):
        # Ollama local client (free, no API key)
        self.model = "llama3.2"

        # ChromaDB local persistent storage (no cloud, no cost)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Use sentence-transformers for free local embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="finsight_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Text splitter for document chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    # ── Document Ingestion ────────────────────────────────────────────────────

    def ingest_text(self, text: str, source_name: str) -> int:
        """Chunk text, embed, and store in ChromaDB."""
        chunks = self.splitter.split_text(text)
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

        # Upsert to avoid duplicates
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Extract text from PDF and ingest."""
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n\n"
        return self.ingest_text(full_text, filename)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Retrieve top-k most relevant chunks from vector DB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count() or 1)
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "relevance_score": round(1 - dist, 3)  # cosine similarity
                })
        return chunks

    # ── LLM Response Generation ───────────────────────────────────────────────

    def answer_with_rag(self, question: str, top_k: int = 4) -> dict:
        """Full RAG pipeline: retrieve + ground + generate."""
        chunks = self.retrieve(question, top_k)

        if not chunks:
            return {
                "answer": "No documents have been ingested yet. Please upload a financial document first.",
                "sources": [],
                "chunks_retrieved": 0,
                "grounding_score": 0.0
            }

        # Build grounded context
        context = "\n\n---\n\n".join([
            f"[Source: {c['source']} | Relevance: {c['relevance_score']}]\n{c['text']}"
            for c in chunks
        ])

        system_prompt = """You are FinSight, a precise financial intelligence assistant.

RULES:
1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, say: "The provided documents do not contain sufficient information to answer this question."
3. Always cite which source document supports each claim.
4. Be concise and factual. No speculation.
5. Format numbers clearly (e.g., $1.2B, 15.3%).

These rules exist to prevent hallucination and ensure all responses are grounded in source documents."""

        user_prompt = f"""Context from financial documents:
{context}

Question: {question}

Provide a grounded answer citing specific sources."""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response["message"]["content"]
        grounding_score = self._compute_grounding_score(answer, chunks)

        return {
            "answer": answer,
            "sources": chunks,
            "chunks_retrieved": len(chunks),
            "grounding_score": grounding_score
        }

    def answer_without_rag(self, question: str) -> dict:
        """Direct LLM answer with NO retrieval (shows hallucination risk)."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial assistant. Answer the question based on your training knowledge."},
                {"role": "user", "content": question}
            ]
        )
        return {
            "answer": response["message"]["content"],
            "sources": [],
            "chunks_retrieved": 0,
            "grounding_score": None
        }

    # ── Hallucination Scoring ─────────────────────────────────────────────────

    def _compute_grounding_score(self, answer: str, chunks: list[dict]) -> float:
        """
        Estimate how grounded an answer is in retrieved chunks.
        Simple heuristic: overlap of key terms between answer and retrieved context.
        In production this would use an NLI model.
        """
        answer_lower = answer.lower()
        all_chunk_text = " ".join([c["text"].lower() for c in chunks])

        # Extract meaningful tokens (filter stopwords)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                     "at", "to", "for", "of", "and", "or", "but", "with", "this",
                     "that", "it", "be", "as", "by", "from", "has", "have", "had"}

        answer_tokens = set(re.findall(r'\b[a-z]{4,}\b', answer_lower)) - stopwords
        chunk_tokens = set(re.findall(r'\b[a-z]{4,}\b', all_chunk_text)) - stopwords

        if not answer_tokens:
            return 1.0

        overlap = answer_tokens & chunk_tokens
        score = len(overlap) / len(answer_tokens)
        return round(min(score, 1.0), 3)

    def get_collection_stats(self) -> dict:
        count = self.collection.count()
        return {"total_chunks": count, "status": "ready" if count > 0 else "empty"}
