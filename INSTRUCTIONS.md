# DocuMind AI - Instructions

## Overview

DocuMind AI is a document reader application with RAG-based question answering, quiz generation, and calculator capabilities. It uses Ollama for local LLM processing, ChromaDB for vector storage, and Tkinter for the GUI.

## Prerequisites

1. **Python 3.9+** installed
2. **Ollama** installed and running locally
3. **Required Ollama models** downloaded

## Setup Instructions

### 1. Install Python Dependencies

```bash
cd DocuMind_AI
pip install -r requirements.txt
```

### 2. Install and Run Ollama

Download Ollama from https://ollama.ai/

Start Ollama in the background:
```bash
ollama serve
```

### 3. Download Required Models

```bash
# Chat model
ollama pull llama3.2

# Embedding model
ollama pull nomic-embed-text
```

Verify models are available:
```bash
ollama list
```

### 4. Run the Application

```bash
cd DocuMind_AI
python -m app.main
```

Or from the project root:
```bash
python -m app.main
```

## Usage

### Loading Documents

1. Click "Load Document" button
2. Select a PDF, DOCX, or TXT file
3. The document will be processed, chunked, and stored in the database
4. A confirmation message will show the number of chunks created

### Asking Questions

1. Type your question in the input field at the bottom
2. Press Enter or click "Send"
3. The bot will retrieve relevant chunks and generate an answer
4. Sources from the retrieved documents are shown below the answer

### Quiz Generation

1. Select a quiz type (Multiple Choice, True/False, or Short Answer)
2. Choose the number of questions (3-10)
3. Click "Generate Quiz"
4. Read through the questions in the chat area
5. Enter your answers in format: `1:A, 2:B, 3:C, 4:D`
6. Click "Submit Quiz"
7. View your score and feedback

### Calculator Agent

If your question contains math expressions (e.g., "what is 25 * 4 + 10?"), the calculator agent will detect it and provide the calculation result.

### Clearing Data

- "Clear Chat": Clears conversation history
- "Clear Database": Removes all documents from the database (requires confirmation)

## Project Structure

```
DocuMind_AI/
├── app/
│   ├── __init__.py          # Package marker
│   ├── main.py              # GUI application entry point
│   ├── models.py            # Ollama model initialization
│   ├── database.py          # ChromaDB + SQLite management
│   ├── document_processor.py # PDF/DOCX loading and chunking
│   ├── chains.py            # LangChain QA and Quiz chains
│   └── prompts.py           # Prompt templates
├── data/                     # Database storage
│   ├── documind.db          # SQLite database
│   └── chroma_db/            # ChromaDB vector store
├── requirements.txt          # Python dependencies
├── INSTRUCTIONS.md          # This file
└── SPEC.md                  # Project specification
```

## Troubleshooting

### Ollama Connection Issues

If you see "Could not connect to Ollama":
1. Ensure Ollama is running: `ollama serve`
2. Check if models are downloaded: `ollama list`
3. Try pulling models again: `ollama pull llama3.2`

### Document Processing Errors

- **Empty chunks**: The document may be scanned/image-based (PDF) or empty
- **Unsupported format**: Only PDF, DOCX, and TXT are supported

### Memory Issues

If the application runs out of memory:
- Reduce the number of chunks retrieved (edit `top_k` in `chains.py`)
- Use smaller documents
- Ensure Ollama isn't using too much memory

### ChromaDB Errors

If you encounter ChromaDB errors:
- Delete the `data/chroma_db` folder
- Restart the application

## Keyboard Shortcuts

- **Enter**: Submit question
- **Ctrl+Delete**: Clear chat

## Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | .pdf | Text extraction works best with text-based PDFs |
| Word | .docx | Full document support including tables |
| Text | .txt | Plain text files |

## Model Configuration

Default models (can be changed in `app/models.py`):
- **Chat**: llama3.2
- **Embedding**: nomic-embed-text

## Database

- **ChromaDB**: Stores vector embeddings for semantic search
- **SQLite**: Stores document metadata and quiz results

## License

This project is for educational purposes.
