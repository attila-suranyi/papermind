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

3. **Configure the application:**

   Configuration is stored in YAML files under the `config/` directory. Two environments are provided out of the box:

   | File | Purpose |
   |------|---------|
   | `config/prod.yaml` | Production settings (default) |
   | `config/test.yaml` | Test settings (used by the integration test suite) |

   Edit the relevant YAML file to change any setting:

   | Key | Description | Default (`prod.yaml`) |
   |-----|-------------|-----------------------|
   | `LLM_BACKEND` | LLM backend to use (`ollama` or `gemini`) | `ollama` |
   | `LLM_MODEL` | Model name for the selected backend | `llama3` |
   | `GEMINI_API_KEY` | API key for Google Gemini (required when using `gemini` backend) | `""` |
   | `EMBEDDER_MODEL` | Sentence-Transformers model for embeddings | `all-MiniLM-L6-v2` |
   | `DOCS_DIR` | Directory where uploaded PDFs are stored | `docs` |
   | `VECTOR_DB_DIR` | Directory for the ChromaDB persistent storage | `vector_db` |
   | `VECTOR_DB_COLLECTION_NAME` | ChromaDB collection name | `pdf_collection` |

   Relative paths in the YAML files are resolved from the project root automatically.

   The active environment is selected at startup via the `--env` flag or the `APP_ENV` environment variable (defaults to `prod`).

## Running the Application

Start the FastAPI server:
```bash
python main.py
```
Or with a specific environment:
```bash
python main.py --env prod
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

### Upload PDF file
```bash
curl -X POST http://127.0.0.1:8000/index \
     -F "file=@path/to/your/document.pdf"
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
- [x] File upload using UploadFile
- [x] Unit tests
- [ ] Evaluation with RAGAS
- [ ] Streaming LLM responses
- [ ] Deployment to cloud
