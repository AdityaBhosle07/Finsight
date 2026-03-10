# FinSight — RAG-Powered Financial Intelligence Assistant

> Ask questions about SEC filings, earnings reports, and financial documents.
> See how RAG eliminates hallucinations compared to raw LLM responses.

Built with: **FastAPI** | **ChromaDB** | **Anthropic Claude** | **Sentence Transformers** | **LangChain**

---

## What This Demonstrates

| Feature | Description |
|---|---|
| Document Chunking | RecursiveCharacterTextSplitter with 800-token chunks, 100-token overlap |
| Embeddings | Local sentence-transformers (all-MiniLM-L6-v2) — no API cost |
| Vector Database | ChromaDB persistent local storage — no cloud required |
| Retrieval Logic | Cosine similarity search, top-k configurable |
| LLM Grounding | Claude grounded strictly in retrieved context, cites sources |
| Hallucination Demo | Side-by-side RAG vs raw LLM — see the difference live |
| Grounding Score | Term overlap heuristic showing how grounded each response is |

---

## Setup (5 minutes)

### 1. Clone and enter project
```bash
git clone <your-repo-url>
cd finsight
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=your_api_key_here
# Windows: set ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Start the backend
```bash
cd backend
python main.py
# API runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 6. Open the frontend
Open `frontend/index.html` in your browser (no server needed).

---

## Using FinSight

### Step 1 — Ingest a document
- Upload a PDF (SEC filing, 10-K, earnings report)
- OR paste the sample text from `sample_docs/acme_q3_2024_earnings.txt`

### Step 2 — Ask a question
Try: *"What was the total revenue reported?"*

### Step 3 — See the magic
Click **Compare RAG vs No-RAG** to see how grounded answers differ from hallucinated ones.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API status and DB stats |
| POST | `/ingest/pdf` | Upload and ingest a PDF |
| POST | `/ingest/text` | Ingest plain text |
| POST | `/query` | Ask a question (RAG or raw) |
| POST | `/compare` | Side-by-side RAG vs non-RAG |
| DELETE | `/reset` | Clear the vector database |

Full interactive docs at `http://localhost:8000/docs`

---

## Architecture

```
User Question
     │
     ▼
Embedding Model (MiniLM-L6-v2)
     │
     ▼
ChromaDB Vector Search ──► Top-K Relevant Chunks
     │
     ▼
Context Assembly + Source Attribution
     │
     ▼
Claude Sonnet (Grounded Prompt)
     │
     ▼
Answer + Sources + Grounding Score
```

---

## Project Structure

```
finsight/
├── backend/
│   ├── main.py          # FastAPI app, all endpoints
│   └── rag_pipeline.py  # Chunking, embeddings, retrieval, LLM grounding
├── frontend/
│   └── index.html       # Single-file UI (no framework needed)
├── sample_docs/
│   └── acme_q3_2024_earnings.txt  # Sample financial document for testing
├── requirements.txt
└── README.md
```

---

## LinkedIn Post Angle

*"I built a RAG-powered financial intelligence assistant that grounds every answer in source documents and shows you exactly where the information came from.*

*The most interesting part: the side-by-side hallucination demo. Ask the same question with RAG on vs off and watch how different the answers are for specific financial figures.*

*Stack: FastAPI, ChromaDB, Anthropic Claude, Sentence Transformers, LangChain.*

*Repo: [link]"*
