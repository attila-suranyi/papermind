import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.ingestion.ingestion_pipeline import IngestionPipeline
from app.retrieval.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    fastapi_app.state.retrieval_pipeline = RetrievalPipeline()
    fastapi_app.state.ingestion_pipeline = IngestionPipeline()
    yield
    fastapi_app.state.retrieval_pipeline = None
    fastapi_app.state.ingestion_pipeline = None


app = FastAPI(title="PaperMind API", lifespan=lifespan)


class IndexRequest(BaseModel):
    docs_dir: Optional[str] = None


class IndexResponse(BaseModel):
    status: str
    docs_dir: Optional[str]


class AnswerRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    query: str
    answer: str


@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}


# TODO use UploadFile for actual file upload
@app.post("/index", response_model=IndexResponse, status_code=202)
def index(req: IndexRequest, request: Request, background_tasks: BackgroundTasks):
    """Trigger indexing of PDFs. Runs indexing in background and returns 202."""
    docs_dir = Path(req.docs_dir) if req.docs_dir else None
    # run the indexing in background to avoid blocking the request
    background_tasks.add_task(request.app.state.ingestion_pipeline.index_pdfs, docs_dir)
    logger.info("Indexing scheduled for %s", docs_dir or "default")

    return IndexResponse(status="ok", docs_dir=str(docs_dir) if docs_dir else None)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request):
    """Get an answer for the provided query using retrieval + LLM."""
    llm_answer = request.app.state.retrieval_pipeline.get_answer(req.query)
    if llm_answer is None:
        raise HTTPException(status_code=500, detail="failed to produce an answer")

    return AnswerResponse(query=req.query, answer=llm_answer)
