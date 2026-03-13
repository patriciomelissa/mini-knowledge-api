from fastapi import APIRouter

from app.schemas.request_response import AskRequest, AskResponse, StateResponse
from app.utils import rag_service

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
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


@router.get("/state", response_model=StateResponse)
def state():
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
