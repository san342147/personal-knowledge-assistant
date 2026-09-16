import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.generation import GroqGenerator
from app.ingest import parse_document
from app.models import Answer, AskRequest
from app.retrieval import Retriever
from app.service import DeskService
from app.telemetry import Telemetry


@asynccontextmanager
async def lifespan(app):
    settings = get_settings()
    app.state.settings = settings
    app.state.telemetry = Telemetry()
    app.state.retriever = Retriever(settings)
    app.state.generator = GroqGenerator(settings)
    app.state.service = DeskService(settings, app.state.retriever, app.state.generator, app.state.telemetry)
    yield


app = FastAPI(title="GroundedDesk", version="1.0.0", lifespan=lifespan,
              description="Local PDF retrieval with cited Groq answers and measurable grounding.")


@app.middleware("http")
async def log_other_requests(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    # Ask has token-aware service logging. Validation failures still get an event.
    if request.url.path != "/ask" or response.status_code == 422:
        telemetry = getattr(request.app.state, "telemetry", None)
        if telemetry:
            telemetry.record({"request_id": uuid.uuid4().hex, "endpoint": request.url.path,
                "status_code": response.status_code, "latency_ms": round((time.perf_counter()-start)*1000, 2),
                "tokens_in": 0, "tokens_out": 0, "estimated_cost_usd": 0, "retrieved_chunk_ids": []})
    return response


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    settings = app.state.settings
    try:
        content = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the 20 MiB limit")
    parsed = await run_in_threadpool(parse_document, file.filename or "unnamed", content,
        settings.chunk_size, settings.chunk_overlap, settings.max_document_chars)
    if not parsed.chunks:
        return JSONResponse(status_code=422, content={"detail": "No text indexed", "warnings": parsed.warnings})
    try:
        added = await run_in_threadpool(app.state.retriever.add, parsed.chunks)
    except Exception as exc:
        raise HTTPException(503, f"Index unavailable ({type(exc).__name__}); check model downloads") from None
    return {"filename": parsed.filename, "document_id": parsed.document_id,
            "chunks_added": added, "chunks_total_in_document": len(parsed.chunks), "warnings": parsed.warnings}


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(422, "Question cannot be blank")
    result = app.state.service.ask(request.question.strip(), request.top_k)
    if result.reason in {"service_error", "generation_unavailable"}:
        return JSONResponse(status_code=503, content=result.model_dump())
    return result


@app.get("/health")
def health():
    try:
        vector = app.state.retriever.health()
    except Exception as exc:
        vector = {"status": "error", "error_type": type(exc).__name__}
    groq = app.state.generator.health()
    ready = vector["status"] == "ok" and groq["status"] == "ok"
    return JSONResponse(status_code=200 if ready else 503,
        content={"status": "ok" if ready else "degraded", "vectorstore": vector, "groq": groq})


@app.get("/metrics")
def metrics():
    return app.state.telemetry.snapshot()
