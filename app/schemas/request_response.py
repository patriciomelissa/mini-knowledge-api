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
