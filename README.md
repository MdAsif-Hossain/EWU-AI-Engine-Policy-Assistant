# 🏆 LLM-Based Agentic AI Engine  
**Champion – EWU Innovation Challenge 2026**

🏅 **Winner of the “LLM-Based AI Engine Development” Innovation Challenge**  
Organized by **East West University Robotics Club**

---

## 📖 Overview

This project is a **fully offline, privacy-first Agentic Retrieval-Augmented Generation (RAG) system** designed to analyze and answer complex policy and regulatory documents with **high factual precision and zero hallucinations**.

Unlike standard chatbots, this system implements an **agentic workflow** capable of reasoning, tool usage, and answer validation through a **multi-stage retrieval pipeline**.

The system was developed to solve a real institutional problem:

> 📘 **Accurately querying the East West University Disciplinary Code**  
> without hallucination, external APIs, or privacy risks.

---

## 🖼️ System Workflow

![System Workflow](./assets/workflow.png)

---

## 🌐 Web Interface

![Web Interface Screenshot](./assets/web_ui.png)

---

## 🚀 Key Features

### 🧠 Agentic Workflow
- Built using **LangGraph**
- Dynamic decision-making for retrieval, reasoning, and rejection

### 🔍 Hybrid Retrieval Pipeline
- Vector Search + Cross-Encoder Re-Ranking
- Strict relevance filtering

### 🔐 Privacy-First & Fully Offline
- Local LLM (**Qwen 2.5 – 1.5B**) via Ollama
- No cloud APIs or data leakage

### 🧩 Semantic Chunking
- Structure-aware PDF ingestion
- Context-preserving chunking

### 🧱 Microservices Architecture
- FastAPI backend
- Streamlit frontend

---

## 🏗️ Architecture

PDFs → Semantic Chunking → Embeddings → ChromaDB → FastAPI → Reranker → Agent → LLM → UI

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **LLM:** Qwen 2.5 (1.5B) via Ollama
- **Embeddings:** all-MiniLM-L6-v2
- **Reranker:** ms-marco-MiniLM-L-6-v2
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Vector DB:** ChromaDB
- **Orchestration:** LangChain, LangGraph

---

## ⚡ Setup

### Prerequisites
```bash
ollama run qwen2.5:1.5b
```

### Installation
```bash
git clone https://github.com/yourusername/ewu-ai-engine.git
cd ewu-ai-engine
pip install -r requirements.txt
```

### Build Knowledge Base
```bash
python ingest.py
```

### Start Backend
```bash
python server.py
```

### Start Frontend
```bash
streamlit run app.py
```

---

## 🧪 Performance

- High retrieval precision via reranking
- CPU-friendly inference
- Hallucination-safe responses

---

## 👤 Author

**Md. Asif Hossain**  
Department of Computer Science and Engineering  
East West University

---

## 📜 License

MIT License
