<div align="center">

<img src="https://img.shields.io/badge/STATUS-LIVE-00ff88?style=for-the-badge&labelColor=0d0d0d" />
<img src="https://img.shields.io/badge/STREAMLIT-CLOUD-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0d0d0d" />
<img src="https://img.shields.io/badge/RENDER-BACKEND-46E3B7?style=for-the-badge&logo=render&logoColor=white&labelColor=0d0d0d" />

<br /><br />

```
██████╗  ██████╗  ██████╗██╗   ██╗███╗   ███╗██╗███╗   ██╗██████╗     █████╗ ██╗
██╔══██╗██╔═══██╗██╔════╝██║   ██║████╗ ████║██║████╗  ██║██╔══██╗   ██╔══██╗██║
██║  ██║██║   ██║██║     ██║   ██║██╔████╔██║██║██╔██╗ ██║██║  ██║   ███████║██║
██║  ██║██║   ██║██║     ██║   ██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║   ██╔══██║██║
██████╔╝╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██████╔╝██╗██║  ██║██║
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝
```

### **AI-Powered Document Intelligence & Analysis System**
*Query any PDF in plain English. Get cited, source-grounded answers in milliseconds.*

<br />

[![Live App](https://img.shields.io/badge/%20Live%20App-docu--mind--intelligence.streamlit.app-FF4B4B?style=for-the-badge)](https://docu-mind-intelligence.streamlit.app)
[![Backend API](https://img.shields.io/badge/%20Backend%20API-documind--ai--dfui.onrender.com-46E3B7?style=for-the-badge)](https://documind-ai-dfui.onrender.com)

</div>

---

## What Is DocuMind?

DocuMind is a **production-deployed RAG (Retrieval-Augmented Generation) system** that lets users interrogate large PDFs using natural language — and receive precise, source-cited answers grounded in actual document content, not hallucinations.

Upload a 50+ page research paper, legal document, or financial report and ask it anything. DocuMind retrieves the right chunks, constructs a grounded prompt, and returns a cited answer in under a second.

---

## Performance Benchmarks

| Operation | Latency |
|-----------|---------|
| PDF Ingestion + Chunking | < 2s per doc |
| Semantic Embedding (384-dim) | Batch processed |
| FAISS Top-4 Vector Retrieval | **~47 ms** |
| Full LLM Response (Groq Llama 3.1) | **~655 ms** |
| Chunks per 15-page document | **100 chunks** |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    Streamlit Cloud Frontend                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                            │
│   PDF Upload → PyMuPDF Parsing → Semantic Text Chunking        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EMBEDDING LAYER                             │
│   Sentence Transformers → 384-dim Dense Vectors → FAISS Index  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL LAYER                             │
│   Query Embedding → FAISS Similarity Search → Top-4 Chunks     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION LAYER                             │
│   Custom Prompt Template → Groq API (Llama 3.1 8B) → Answer   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM Backend** | Groq API — Llama 3.1 8B |
| **Embeddings** | Sentence Transformers (384-dim) |
| **Vector Store** | FAISS |
| **Orchestration** | LangChain |
| **PDF Parsing** | PyMuPDF |
| **Frontend** | Streamlit |
| **Backend** | Render |
| **Language** | Python |

---

## Core Features

- **50+ page PDF support** — handles large documents without truncation
- **Semantic chunking** — context-aware splitting, not naive character splits
- **FAISS vector indexing** — millisecond nearest-neighbor retrieval
- **Custom Groq prompts** — engineered templates that reduce hallucinations
- **Session-state conversation history** — multi-turn Q&A within a session
- **Source-cited answers** — every response grounded in retrieved document chunks
- **Dual deployment** — Streamlit Cloud (UI) + Render (backend API)

---

## Run Locally

```bash
git clone https://github.com/sujanya-hub/DocuMind-AI-Powered-Document-Intelligence-Analysis-System
cd DocuMind-AI-Powered-Document-Intelligence-Analysis-System
pip install -r requirements.txt
```

Add your API keys to `.env`:
```env
GROQ_API_KEY=your_groq_api_key
```

```bash
streamlit run app.py
```

---

## Project Structure

```
DocuMind/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── ingestion.py        # PDF parsing & chunking (PyMuPDF)
│   ├── embeddings.py       # Sentence Transformer embedding
│   ├── retrieval.py        # FAISS indexing & search
│   └── generation.py       # Groq prompt templates & LLM calls
├── requirements.txt
└── .env.example
```

---

## Live Deployments

| Service | URL |
|---------|-----|
| **Streamlit App** | [docu-mind-intelligence.streamlit.app](https://docu-mind-intelligence.streamlit.app) |
| **Render Backend** | [documind-ai-dfui.onrender.com](https://documind-ai-dfui.onrender.com) |

> *Render free-tier has a cold-start delay of ~7s on first request. Subsequent requests are fast.*

---

<div align="center">

**Built by [Sujanya Srinivas](https://linkedin.com/in/sujanya-s-538a7a2b1)**
[LinkedIn](https://linkedin.com/in/sujanya-s-538a7a2b1) · [GitHub](https://github.com/sujanya-hub) · [Email](mailto:sujanyasrinivasa@gmail.com)

</div>
