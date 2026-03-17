from fastapi import APIRouter

from app.schemas.request_response import (
    AskRequest,
    AskResponse,
    RetrievalResponse,
    StateResponse,
)
from app.utils import rag_service

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Handle question answering request using the RAG service pipeline.

    Args:
        request (AskRequest): Request object containing the user question.

    Returns:
        AskResponse: Response containing the generated answer and supporting sources.
    """
    try:
        result = rag_service.ask(request.question)

    except Exception:
        result = {"answer": "", "sources": []}

    return AskResponse(answer=result["answer"], sources=result["sources"])


@router.post("/debug/retrieve", response_model=RetrievalResponse)
def debug_retrieve(request: AskRequest) -> RetrievalResponse:
    """
    Debug endpoint for inspecting raw retrieval results.

    This endpoint allows testing and validating the retrieval step of the
    RAG pipeline independently from reranking and LLM generation.

    Behavior:
        - Ensures the RAG service is initialized.
        - Executes vector similarity search using the input question.
        - Returns retrieved chunks with metadata and similarity scores.
        - In case of failure, returns a default empty-like result structure.

    Args:
        request (AskRequest): Request containing the user query.

    Returns:
        RetrievalResponse: Object containing:
            - question (str): Input query.
            - results (List[Dict[str, Any]]): Retrieved document chunks with
              metadata (document, page, chunk_id, score, text).
    """
    try:
        rag_service.ensure_initialized()

        results = rag_service.retrieve(request.question)
    except Exception:
        results = [{"document": "", "page": 0, "chunk_id": 0, "score": 0.0, "text": ""}]

    return RetrievalResponse(question=request.question, results=results)


@router.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    """
    State check endpoint for the RAG service.

    This endpoint provides basic runtime diagnostics about the system,
    including the status of the API and information about the vector
    index currently loaded in memory.

    It is typically used for monitoring, deployment checks, or
    orchestration readiness probes.

    Returns:
        StateResponse: Object containing:
            - status (str): Service health status indicator.
            - vector_index_size (int): Number of vectors currently stored
              in the FAISS index.
            - documents_indexed (int): Total number of document chunks
              stored in the vector store metadata.
    """
    try:
        rag_service.ensure_initialized()

        message = "ok"
        index_size = rag_service.vector_store.index.ntotal
        docs = len(rag_service.vector_store.metadata)

    except Exception:
        message = "Not ok"
        index_size = 0
        docs = 0

    return StateResponse(
        status=message,
        vector_index_size=index_size,
        documents_indexed=docs,
    )
