# EWU AI Engine: Policy Assistant (Agentic RAG)

A **Retrieval-Augmented Generation (RAG)** assistant for **East West University (EWU)** policy/Q&A. The system ingests EWU policy PDFs, indexes them in a vector database, retrieves the most relevant clauses for a question, re-ranks the candidates, and generates an answer **strictly grounded in the retrieved policy text**.

> UI: Streamlit chat app  
> Tool/Backend: FastAPI “tool server” for retrieval + utility tools  
> LLM runtime: `llama-cpp-python` running a local GGUF model (`qwen3b.gguf`)

---

## Screenshot (Home Page)

Create an `assets/` folder and add your screenshot image there.

- **Recommended file name (clean):** `assets/home.png`  
- **Your screenshot name:** `Screenshot 2026-01-26 220440.png`

If you keep your original name, use:

![EWU AI Engine - Home Page](assets/Screenshot%202026-01-26%20220440.png)

If you rename it to `home.png` (recommended), use:

![EWU AI Engine - Home Page](assets/home.png)

> GitHub requires spaces in image paths to be URL-encoded (`%20`). Renaming avoids broken links.

---

## System Diagram

```mermaid
flowchart LR
  U[User] -->|Question| S[Streamlit UI (app.py)]

  subgraph AG[LangGraph Agent (StateGraph)]
    P[Preprocess / Keyword Extraction\n(Qwen via llama.cpp)] --> R[Retrieve Context\nPOST /search]
    R --> PL[Planner\n(optional tool decision)]
    PL -->|if needed| C[Calculator Tool\nPOST /calculate]
    PL -->|otherwise| G[Generate Answer\nStrict Grounding]
    C --> G
  end

  S --> P
  G --> S

  subgraph TS[Tool Server (FastAPI - server.py)]
    A1[/search endpoint/] --> V[(Chroma Vector DB\nchroma_db/)]
    A1 --> RR[Cross-Encoder Reranker\nms-marco-MiniLM-L-6-v2]
    A2[/calculate endpoint/]
  end

  R --> A1
  C --> A2

  subgraph ING[Ingestion (ingest.py)]
    PDF[data/pdfs/*.pdf] --> L[PyPDFLoader]
    L --> T[Chunking\nRecursiveCharacterTextSplitter]
    T --> E[Embeddings\nall-MiniLM-L6-v2]
    E --> V
  end
```

---

## Key Features

- **Grounded answers (anti-hallucination):** The LLM is instructed to use *only* the retrieved context. If the answer is not present, it responds:  
  **“I don't know based on the provided documents.”**
- **Agentic workflow (LangGraph):** A multi-step graph pipeline:
  1. Query preprocessing (keyword extraction)
  2. Document retrieval from vector DB (Chroma)
  3. Re-ranking with a cross-encoder
  4. Tool-planning (optional calculator tool)
  5. Answer generation with strict grounding
- **PDF ingestion pipeline:** Loads policy PDFs, chunks them, embeds chunks, and persists them into Chroma.
- **Re-ranking for better relevance:** Uses a cross-encoder reranker to prioritize the best clauses before sending context to the LLM.
- **Transparency:** The Streamlit UI shows a “reasoning chain” (step log) and the source filenames used.

---

## Architecture (High Level)

1. **Ingestion (`ingest.py`)**
   - Load PDFs from `data/pdfs/` using `PyPDFLoader`
   - Split into chunks using `RecursiveCharacterTextSplitter`  
     - `chunk_size=600`, `chunk_overlap=100`
   - Embed each chunk with `sentence-transformers/all-MiniLM-L6-v2`
   - Store embeddings + metadata in **Chroma** persisted at `chroma_db/`

2. **Tool Server (`server.py`)**
   - Exposes retrieval endpoint `POST /search`
     - Initial retrieval: `vector_db.similarity_search(query, k=25)`
     - Re-rank retrieved chunks using: `cross-encoder/ms-marco-MiniLM-L-6-v2`
     - Returns top results (typically top 5)
   - Exposes calculator endpoint `POST /calculate` (example tool)

3. **Agent + UI (`app.py`)**
   - Streamlit chat UI
   - Loads a local GGUF model via `llama_cpp.Llama`:
     - `models/qwen3b.gguf`
     - `n_ctx=2048`, `n_gpu_layers=25`
   - LangGraph nodes:
     - **preprocess**: extract retrieval keywords (few-shot prompt tuned for policy terms)
     - **retrieve**: call the tool server `/search` to fetch relevant policy clauses
     - **reason**: decide whether to call calculator tool
     - **tool**: optional `/calculate`
     - **generate**: answer strictly from retrieved context

---

## Technology Stack

### Core
- **Python**
- **Streamlit** — interactive chat UI
- **FastAPI + Uvicorn** — tool server (retrieval + utilities)
- **Requests** — client calls from the UI agent to the tool server

### RAG / Orchestration
- **LangGraph** — graph-based agent workflow (state machine style)
- **LangChain Community**
  - `PyPDFLoader` — PDF text extraction
  - `RecursiveCharacterTextSplitter` — chunking strategy
  - `HuggingFaceEmbeddings` — embedding wrapper
  - `Chroma` vector store integration

### Vector Database
- **ChromaDB (Chroma)** — persistent local vector store (`chroma_db/`)

### Embeddings & Re-ranking Models
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder)

### LLM Inference
- **llama-cpp-python (`llama_cpp`)** running a local **GGUF** model:
  - `models/qwen3b.gguf`

---

## Repository Structure (Expected)

```
.
├── app.py
├── server.py
├── ingest.py
├── data/
│   └── pdfs/
├── chroma_db/          # generated
├── models/
│   └── qwen3b.gguf
└── assets/
    └── Screenshot 2026-01-26 220440.png
```

---

## Setup & Installation

### 1) Create environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -U streamlit fastapi uvicorn requests
pip install -U langgraph langchain-community langchain-text-splitters
pip install -U chromadb sentence-transformers
pip install -U llama-cpp-python
```

---

## Usage (End-to-End)

### Step 1 — Add policy PDFs
Put EWU policy PDFs into:
```bash
data/pdfs/
```

### Step 2 — Build the vector database
```bash
python ingest.py
```

### Step 3 — Start the tool server
```bash
python server.py
# or:
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Step 4 — Run the Streamlit app
```bash
streamlit run app.py
```

---

## Troubleshooting

- **Database folder not found**
  - Ensure you ran: `python ingest.py`
- **Tool server unreachable**
  - Confirm `server.py` is running and `TOOL_URL` in `app.py` matches
- **No documents found**
  - Ensure PDFs exist in `data/pdfs/` and contain readable text
- **Model file not found**
  - Ensure `models/qwen3b.gguf` exists and `MODEL_PATH` is correct

---

## License
Add your license here (MIT / Apache-2.0 / etc.) and clarify whether policy PDFs are redistributed or only used locally.
