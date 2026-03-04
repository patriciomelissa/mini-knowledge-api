from typing import List

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import parameters


class EmbeddingServiceOpenAI:

    def __init__(self):
        self.client = OpenAI(api_key=parameters.OPENAI_API_KEY)
        self.model = "text-embedding-3-small"  # modelo estável e barato

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class EmbeddingService:

    def __init__(self):
        # Modelo leve e estável
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
