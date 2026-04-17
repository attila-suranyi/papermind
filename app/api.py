import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from app.embedder import Embedder
from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.llm import GeminiClient, LLMClient, OllamaClient
from app.retrieval.retrieval_pipeline import RetrievalPipeline
from app.store.chroma_db import ChromaDB
from config import Settings, load_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.settings = load_settings()
    app_settings: Settings = fastapi_app.state.settings
    logger.info(
        "Starting app with settings: %s", app_settings.model_dump(exclude={"GEMINI_API_KEY"})
    )

    db = ChromaDB(
        db_path=app_settings.VECTOR_DB_DIR, collection_name=app_settings.VECTOR_DB_COLLECTION_NAME
    )
    model = SentenceTransformer(app_settings.EMBEDDER_MODEL)
    embedder = Embedder(model)

    llm: LLMClient
    if app_settings.LLM_BACKEND == "gemini":
        llm = GeminiClient(
            default_model=app_settings.LLM_MODEL, api_key=app_settings.GEMINI_API_KEY
        )
    else:
        llm = OllamaClient(default_model=app_settings.LLM_MODEL)

    fastapi_app.state.retrieval_pipeline = RetrievalPipeline(
        db=db, embedder=embedder, llm=llm, llm_model=app_settings.LLM_MODEL
    )
    fastapi_app.state.ingestion_pipeline = IngestionPipeline(
        db=db, embedder=embedder, docs_dir=app_settings.DOCS_DIR
    )
    yield
    # Cleanup if necessary
    fastapi_app.state.retrieval_pipeline = None
    fastapi_app.state.ingestion_pipeline = None


app = FastAPI(title="PaperMind API", lifespan=lifespan)


class IndexRequest(BaseModel):
    docs_dir: Optional[str] = None


class IndexResponse(BaseModel):
    status: str
    filename: Optional[str]


class AnswerRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    query: str
    answer: str


@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse, status_code=202)
async def index(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and trigger indexing. Runs indexing in background and returns 202."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    app_settings: Settings = request.app.state.settings
    docs_dir = app_settings.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    file_path = docs_dir / file.filename

    # Save the file
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # run the indexing in background to avoid blocking the request
    background_tasks.add_task(request.app.state.ingestion_pipeline.index_pdf, file_path)
    logger.info("Indexing scheduled for %s", file.filename)

    return IndexResponse(status="ok", filename=file.filename)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request):
    """Get an answer for the provided query using retrieval + LLM."""
    llm_answer = request.app.state.retrieval_pipeline.get_answer(req.query)
    if llm_answer is None:
        raise HTTPException(status_code=500, detail="failed to produce an answer")

    return AnswerResponse(query=req.query, answer=llm_answer)
