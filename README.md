# ⚡ DocMind AI

> **RAG-Powered Document Intelligence** — Ask questions about your documents and get precise, context-grounded answers using a fully local AI pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)

---

## 📖 Overview

DocMind AI is a desktop application that brings **Retrieval-Augmented Generation (RAG)** to your local machine. Upload any document, and instantly have a conversation with it — powered entirely by local LLMs via [Ollama](https://ollama.com), with no data ever leaving your device.

The pipeline handles everything: document loading → intelligent chunking → vector embedding → semantic retrieval → answer generation — all surfaced through a clean, dark-themed GUI.

---

## ✨ Features

- 🗂 **Multi-format support** — PDF, TXT, CSV, DOCX, and Markdown
- 🧠 **Local LLM inference** via Ollama (default: `minimax-m2.7:cloud`)
- 🔍 **Semantic search** with ChromaDB vector store
- ✂️ **Configurable chunking** — tune chunk size, overlap, and retrieval Top-K from the UI
- 📎 **Source attribution** — every answer cites which part of the document it came from
- 🔄 **Hot-swap models** — switch chat models without re-indexing
- 🖥️ **Fully local** — no API keys, no cloud, no data leakage
- 💬 **Threaded UI** — document processing and queries run in background threads, keeping the interface responsive

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
│   Document  │───▶│  Text Chunks │───▶│  OllamaEmbed  │───▶│   ChromaDB     │
│  (any fmt)  │    │  (splitter)  │    │  (qwen3-emb)  │    │  (vector store)│
└─────────────┘    └──────────────┘    └───────────────┘    └───────┬────────┘
                                                                     │  similarity
                                                                     │  search
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────▼────────┐
│   Answer    │◀───│  ChatOllama  │◀───│  RAG Prompt   │◀───│  Top-K Chunks  │
│  + Sources  │    │  (LLM)       │    │  (template)   │    │  (retrieved)   │
└─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘
```

**Stack:** LangChain · Ollama · ChromaDB · Tkinter

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running
- The required models pulled locally:

```bash
ollama pull minimax-m2.7:cloud
ollama pull qwen3-embedding:0.6b
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/docmind-ai.git
cd docmind-ai

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python app.py
```

---

## 🖥️ Usage

1. **Upload a document** — Click **＋ Upload Document** and select a file (PDF, DOCX, TXT, CSV, or Markdown).
2. **Configure chunking** — Adjust *Chunk Size*, *Chunk Overlap*, and *Top-K Results* sliders to your preference.
3. **Process the document** — Click **⚡ Process Document**. The pipeline will load, split, embed, and index your file.
4. **Ask questions** — Type your question in the input box and press **Enter** (or click **Send ➤**).
5. **Review answers** — Responses appear in the chat panel with cited source references.

> **Tip:** Use `Shift + Enter` to insert a newline in the query input without sending.

---

## ⚙️ Configuration

All runtime settings are adjustable from the sidebar — no config file editing needed.

| Setting | Default | Description |
|---|---|---|
| **Chat Model** | `minimax-m2.7:cloud` | Ollama model used for answer generation |
| **Embed Model** | `qwen3-embedding:0.6b` | Ollama model used to embed document chunks |
| **Chunk Size** | `800` | Max characters per text chunk |
| **Chunk Overlap** | `150` | Overlap between adjacent chunks (preserves context) |
| **Top-K Results** | `4` | Number of chunks retrieved per query |

To use a different Ollama model, pull it first and then select it from the **Chat Model** dropdown in the header.

---

## 📁 Project Structure

```
docmind-ai/
├── app.py              # Tkinter GUI — main entry point
├── rag_engine.py       # RAG pipeline (load → chunk → embed → retrieve → generate)
├── requirements.txt    # Python dependencies
├── chroma_db/          # Persisted vector store (auto-created on first index)
└── README.md
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-core` | RAG orchestration & LCEL chain |
| `langchain-community` | Document loaders (PDF, DOCX, CSV, MD) |
| `langchain-ollama` | Ollama embeddings & chat model integration |
| `langchain-chroma` | ChromaDB vector store integration |
| `chromadb` | Local vector database |
| `pypdf` | PDF text extraction |
| `docx2txt` / `python-docx` | Word document support |
| `unstructured` | Markdown & fallback document parsing |

---

## 🗺️ Roadmap

- [ ] Multi-document indexing (query across a folder)
- [ ] Conversation memory across sessions
- [ ] Export chat history to PDF / Markdown
- [ ] Web UI alternative (FastAPI + React)
- [ ] Support for additional Ollama models in the dropdown
- [ ] OpenAI / Anthropic API backend option

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss your idea before submitting a pull request.

```bash
# Run with verbose logging for development
PYTHONPATH=. python app.py
```

---

---

<p align="center">Built with ❤️ using LangChain, Ollama, and ChromaDB</p>
