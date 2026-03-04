import os

from dotenv import load_dotenv

load_dotenv()

class Parameters:
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