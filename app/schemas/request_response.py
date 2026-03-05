from typing import Any, Dict, List

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str = "How many sections is the exam divided into?"


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
