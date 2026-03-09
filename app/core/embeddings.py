from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import parameters


class EmbeddingServiceOpenAI:
    """
    Service responsible for generating text embeddings using OpenAI embedding models.

    This class provides methods to convert text into embedding vectors using
    OpenAI's embedding API. It supports both single text and batch document
    embedding generation.

    Attributes:
        client (OpenAI): OpenAI API client instance.
        model (str): Name of the embedding model used.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=parameters.OPENAI_API_KEY)
        self.model = "text-embedding-3-small"  # modelo estável e barato

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text (str): Input text to be embedded.

        Returns:
            List[float]: Embedding vector representation of the input text.
        """
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text documents.

        Args:
            texts (List[str]): List of input text documents.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class EmbeddingService:
    """
    Service responsible for generating vector embeddings from text.

    This class uses a sentence-transformer model to convert text into numerical
    vector representations (embeddings), enabling applications such as semantic
    search, clustering, or text similarity analysis.

    Attributes:
        model (SentenceTransformer): Embedding model used to encode text.

    Methods:
        embed_text(text):
            Generates the embedding of a single string.

        embed_documents(texts):
            Generates embeddings for a list of documents.
    """

    def __init__(self) -> None:
        # Modelo leve e estável
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text (str): Input text to be embedded.

        Returns:
            List[float]: Embedding vector representation of the input text.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts (List[str]): List of input texts.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
