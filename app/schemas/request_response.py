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


class RetrievalResult(BaseModel):
    """
    Model representing a single retrieval result.

    This model contains the metadata and similarity score for a
    document chunk retrieved from the vector store.

    Attributes:
        document (str): Source document filename.
        page (int): Page number where the chunk was extracted.
        chunk_id (int): Identifier of the chunk within the document.
        score (float): Similarity score (distance) between the query
            and the retrieved chunk.
        text (str): Text content of the retrieved chunk.
    """

    document: str
    page: int
    chunk_id: int
    score: float
    text: str


class RetrievalResponse(BaseModel):
    """
    Response model for retrieval debug endpoint.

    This model encapsulates the user query and the list of retrieved
    document chunks returned by the retrieval step of the RAG pipeline.

    Attributes:
        question (str): Input query used for retrieval.
        results (List[RetrievalResult]): List of retrieved results with metadata.
    """

    question: str
    results: list[RetrievalResult]
