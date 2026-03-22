# DocuMind AI - Document Reader with RAG and Quiz Generation

## 1. Project Overview

**Project Name:** DocuMind AI
**Type:** Desktop Application (Python)
**Core Functionality:** A document reader that stores documents in a database, creates embeddings for semantic search, answers user questions via RAG, and generates quizzes from document content.

## 2. Technology Stack

| Component | Technology |
|-----------|------------|
| Chat Model | Ollama (local) - llama3.2 or similar |
| Embedding Model | Ollama - nomic-embed-text |
| Vector Database | ChromaDB (embeddings + original text) |
| Metadata Database | SQLite |
| Framework | LangChain |
| GUI | Tkinter (built-in) |
| Document Parsing | PyPDF2, python-docx |

## 3. Architecture

```
User Input (Document/Question/Quiz Request)
           │
           ▼
    ┌──────────────────┐
    │   GUI (Tkinter)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Document        │
    │  Processor       │  ──► Splits docs into chunks
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  ChromaDB        │  ◄──► Stores embeddings + chunks
    │  Vector Store    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  LangChain Chain │
    │  - QA Chain      │  ◄──► Ollama LLM
    │  - Quiz Chain    │
    │  - Calculator    │  (Math agent for calculations)
    └──────────────────┘
```

## 4. Functionality Specification

### 4.1 Document Management
- **Load Document:** Support PDF and DOCX files
- **Chunking:** Split documents into 500-token chunks with 50-token overlap
- **Embedding:** Generate embeddings using Ollama (nomic-embed-text)
- **Storage:** Store in ChromaDB with document ID, chunk index, source path

### 4.2 Question Answering (RAG)
- **Process:** Receive question → embed question → retrieve relevant chunks → generate answer
- **Chain Type:** RetrievalQA with ConversationalRetrievalChain
- **Memory:** Chat memory for conversation context
- **Source Display:** Show which document chunks were used

### 4.3 Quiz Generation
- **Quiz Types:** Multiple choice, True/False, Short answer
- **Process:** Analyze document content → generate questions → store in memory
- **Scoring:** Compare user answers to model answers

### 4.4 Calculator Agent
- **Trigger:** Detect math expressions in questions
- **Tool:** LangChain calculator (math agent)
- **Example:** "Calculate the total cost if each item is $19.99 for 5 items"

### 4.5 Chat Interface
- **Conversation History:** Track user-bot dialogue
- **Clear Context:** Option to reset conversation
- **Document Reference:** Show which document is active

## 5. Database Schema

### ChromaDB Collections
- **collection_name:** `document_chunks`
- **documents:** Original text chunks
- **embeddings:** Vector embeddings
- **metadatas:** {doc_id, chunk_index, source_file, created_at}

### SQLite Tables
- **documents:** id, filename, filepath, created_at, chunk_count
- **quiz_results:** id, doc_id, score, total_questions, timestamp

## 6. File Structure

```
DocuMind_AI/
├── SPEC.md
├── INSTRUCTIONS.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py           # Tkinter GUI
│   ├── database.py       # ChromaDB + SQLite operations
│   ├── document_processor.py  # PDF/DOCX loading, chunking
│   ├── chains.py         # LangChain chains (QA, Quiz, Calculator)
│   ├── models.py         # Ollama model initialization
│   └── prompts.py        # Prompt templates
└── data/
    ├── chroma_db/        # ChromaDB persistence
    └── documind.db       # SQLite database
```

## 7. GUI Layout

```
┌─────────────────────────────────────────────────────────┐
│  DocuMind AI                              [Clear Chat]  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │ Document Panel                                   │    │
│  │ [Load Document] [Current: document.pdf]         │    │
│  │ [Documents in DB: 3] [Clear DB]                  │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │                                                  │    │
│  │  Chat History / Quiz Display                     │    │
│  │                                                  │    │
│  │                                                  │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │ Quiz Mode: [None ▼]  Questions: [5 ▼]           │    │
│  │ [Generate Quiz] [Submit Quiz]                    │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │ Enter your question...              [Send]      │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 8. User Interactions

1. **Load Document:** Click button → File dialog → Process → Store in DB
2. **Ask Question:** Type in input → Press Enter/Send → Get RAG answer
3. **Generate Quiz:** Select type → Click Generate → View questions
4. **Take Quiz:** Submit answers → Get score and feedback
5. **Clear Chat:** Clear conversation history
6. **Clear DB:** Remove all documents from database

## 9. Error Handling

- File not found: Show error message in chat
- Ollama not running: Prompt to start Ollama
- Empty document: Skip and notify user
- No documents in DB: Prompt to load document first
- Invalid quiz input: Validate and show error

## 10. Acceptance Criteria

- [ ] User can load PDF and DOCX documents
- [ ] Documents are chunked and stored in ChromaDB with embeddings
- [ ] User can ask questions and receive contextually relevant answers
- [ ] Calculator agent handles math expressions in questions
- [ ] User can generate quizzes (multiple choice, T/F, short answer)
- [ ] Quiz results show score and feedback
- [ ] Chat history is maintained during session
- [ ] GUI is responsive and intuitive
- [ ] Application works offline with Ollama
