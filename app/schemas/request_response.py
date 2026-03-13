from typing import Any, Dict, List

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    Request model for question answering API.

    Attributes:
        question (str): User question to be answered.
    """

    question: str = "How many sections is the exam divided into?"


class AskResponse(BaseModel):
    """
    Response model for question answering API.

    Attributes:
        answer (str): Generated answer from the system.
        sources (List[Dict[str, Any]]): List of retrieval sources supporting the answer.
    """

    answer: str
    sources: List[Dict[str, Any]]


class StateResponse(BaseModel):
    """
    Response model for the state check endpoint.

    This model represents the operational status of the RAG service
    along with basic statistics about the vector store currently
    loaded in memory.

    Attributes:
        status (str): Status of the service.
        vector_index_size (int): Number of vectors stored in the FAISS index.
        documents_indexed (int): Total number of document chunks indexed
            in the vector store metadata.
    """

    status: str
    vector_index_size: int
    documents_indexed: int
