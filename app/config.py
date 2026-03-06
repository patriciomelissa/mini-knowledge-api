import os

from dotenv import load_dotenv

load_dotenv()


class Parameters:
    """
    Application configuration parameters loaded from environment variables.

    This class centralizes configuration settings for:
        - Language model integration
        - Retrieval-Augmented Generation (RAG) pipeline
        - Vector database search behavior
        - File system paths

    Attributes:
        OPENAI_API_KEY (str): API key for OpenAI services.
        MODEL_NAME (str): Name of the language model used.
        CHUNK_SIZE (int): Size of text chunks for document processing.
        CHUNK_OVERLAP (int): Overlap size between consecutive text chunks.
        TOP_K (int): Number of top search results to retrieve.
        MAX_DISTANCE (float): Maximum allowed vector distance for retrieval filtering.
        DATA_PATH (str): Path to source documents.
        VECTOR_STORE_PATH (str): Path where vector index and metadata are stored.
    """

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # RAG
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 900))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    TOP_K: int = int(os.getenv("TOP_K", 3))

    # FAISS
    MAX_DISTANCE: float = 2

    # Paths
    DATA_PATH: str = os.getenv("DATA_PATH", "./data")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")


parameters = Parameters()
