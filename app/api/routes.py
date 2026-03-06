from fastapi import APIRouter

from app.schemas.request_response import AskRequest, AskResponse
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
    result = rag_service.ask(request.question)

    return AskResponse(answer=result["answer"], sources=result["sources"])
