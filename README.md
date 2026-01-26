# 🏆 LLM-Based Agentic AI Engine

> **Champion – EWU Innovation Challenge 2026** > 🏅 **Winner** of the *“LLM-Based AI Engine Development” Innovation Challenge* > Organized by **East West University Robotics Club**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![Framework](https://img.shields.io/badge/Framework-LangGraph-orange?style=flat)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5_(Local)-purple?style=flat)
![Status](https://img.shields.io/badge/Status-Maintained-green?style=flat)

---

## 📖 Overview

This project is a **fully offline, privacy-first Agentic Retrieval-Augmented Generation (RAG) system** designed to analyze and answer complex policy and regulatory documents with **high factual precision and zero hallucinations**.

Unlike standard chatbots, this system implements an **agentic workflow** capable of reasoning, tool usage, and answer validation through a **multi-stage retrieval pipeline**.

The system was developed to solve a real institutional problem:

> 📘 **Accurately querying the East West University Disciplinary Code** > *Goal: To provide answers without hallucination, external APIs, or privacy risks.*

---

## 🖼️ System Workflow

![System Workflow](./workflow.png)

---

## 🌐 Web Interface

![Web Interface Screenshot](./Screenshot%202026-01-26%20220440.png)

---

## 🚀 Key Features

### 🧠 Agentic Workflow
- Built using **LangGraph**.
- Dynamic decision-making for retrieval, reasoning, and rejection.

### 🔍 Hybrid Retrieval Pipeline
- **Vector Search** + **Cross-Encoder Re-Ranking**.
- Strict relevance filtering to ensure accuracy.

### 🔐 Privacy-First & Fully Offline
- Powered by **Local LLM (Qwen 2.5 – 1.5B)** via Ollama.
- Zero data leakage; no cloud APIs used.

### 🧩 Semantic Chunking
- Structure-aware PDF ingestion.
- Context-preserving chunking for better retrieval.

### 🧱 Microservices Architecture
- **FastAPI** backend for scalable API handling.
- **Streamlit** frontend for user interaction.

---

## 🏗️ Architecture

```mermaid
graph LR
A[PDFs] --> B(Semantic Chunking)
B --> C(Embeddings)
C --> D[(ChromaDB)]
D --> E[FastAPI]
E --> F{Reranker}
F --> G[Agent Router]
G --> H[Local LLM]
H --> I[UI Response]
