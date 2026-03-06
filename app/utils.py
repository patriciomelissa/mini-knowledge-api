from app.core.rag import RAGService

rag_service = RAGService()


def get_rag_service():
    """
    Retrieve the singleton instance of the RAG service.

    Returns:
        RAGService: Shared RAGService instance used across the application.
    """
    return rag_service
