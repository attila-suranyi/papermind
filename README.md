# PaperMind

PaperMind is a RAG (Retrieval-Augmented Generation) application designed for efficient indexing and querying of PDF documents. It leverages FastAPI for the API layer, Docling for PDF ingestion, and ChromaDB for vector storage.

## Features

- **PDF Ingestion:** Uses Docling for robust PDF parsing.
- **Vector Storage:** ChromaDB for similarity-based document retrieval.
- **Multi-Backend LLM Support:** Works with Ollama (local) or Google Gemini (hosted).
- **Asynchronous Indexing:** Background tasks for document processing.

## Tech Stack

- **Language:** Python 3.12+
- **Framework:** FastAPI
- **Package Manager:** [uv](https://github.com/astral-sh/uv)
- **Vector DB:** ChromaDB
- **PDF Parsing:** Docling
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)

## Requirements

- Python 3.12 or higher.
- (Optional) [Ollama](https://ollama.com/) for local LLM execution.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd papermind
   ```

2. **Install dependencies:**
   Using `uv` (recommended):
   ```bash
   uv sync
   ```

3. **Configure environment variables:**

Create a `.env` file in the root directory:

| Variable | Description | Default                                                 | Required    |
|----------|-------------|---------------------------------------------------------|-------------|
| `LLM_BACKEND` | LLM backend to use (`ollama` or `gemini`) | `ollama`                                                | No          |
| `LLM_MODEL` | Model name for the selected backend | llama3 for local, and gemini-3-flash-preview for gemini | No          |
| `GEMINI_API_KEY` | API key for Google Gemini | -                                                       | Conditional |
| `HOST` | Host to bind the server to | `0.0.0.0`                                               | No          |
| `PORT` | Port to run the server on | `8000`                                                  | No          |

## Running the Application

Start the FastAPI server:
```bash
python main.py
```
Or using `uv`:
```bash
uv run python main.py
```

The API will be available at `http://localhost:8000`.

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/
```

### Trigger PDF Indexing
Indexes documents from the `docs/` directory by default.
```bash
curl -X POST http://localhost:8000/index \
     -H "Content-Type: application/json" \
     -d '{"docs_dir": "docs"}'
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/answer \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic of the paper?"}'
```

## TODO

- [x] PDF Ingestion
- [x] Chunking
- [x] Embedding
- [x] Vector storage
- [x] Retrieve data using similarity search
- [x] Prompt construction
- [x] Pass data to LLM
- [x] API endpoint with FastAPI
- [ ] File upload using UploadFile
- [ ] Unit tests
- [ ] Evaluation with RAGAS
- [ ] Streaming LLM responses
- [ ] Deployment to cloud
