import os
from typing import Optional

from app.config import parameters
from app.core.embeddings import EmbeddingService
from app.core.llm_local import LocalLLMService
from app.core.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor


class RAGService:

    def __init__(self):
        self.embedder = EmbeddingService()
        self.processor = DocumentProcessor()
        self.vector_store: Optional[VectorStore] = None
        self.is_initialized = False
        self.llm = LocalLLMService()

    # -----------------------------
    # PUBLIC API
    # -----------------------------

    def ask(self, question: str) -> dict:
        self._ensure_initialized()

        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(query_embedding)

        # Filtrar por threshold
        filtered_results = [
            (chunk, score)
            for chunk, score in results
            if score <= parameters.MAX_DISTANCE
        ]

        # Se nada relevante
        if not filtered_results:
            return {
                "answer": "I could not find relevant information in the documents.",
                "sources": [],
            }

        # Construir contexto apenas com relevantes
        context_chunks = [r[0] for r in filtered_results]
        context = "\n\n".join(context_chunks)

        # 🔥 Só agora chamamos o LLM
        answer = self.llm.generate_answer(context, question)

        return {"answer": answer, "sources": ["local_documents"]}

    def reindex(self):
        """
        Force rebuild of the index.
        """
        self._create_index()
        self.is_initialized = True

    # -----------------------------
    # INTERNAL LOGIC
    # -----------------------------

    def initialize(self):
        """
        Safe initialization. Can be called multiple times.
        """
        if self.is_initialized:
            return

        if self._index_exists():
            self._load_index()
        else:
            self._create_index()

        self.is_initialized = True

    def _ensure_initialized(self):
        """
        Lazy initialization safeguard.
        """
        if not self.is_initialized:
            self.initialize()

    def _index_exists(self) -> bool:
        index_path = os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss")
        return os.path.exists(index_path)

    def _create_index(self):
        print("Creating new vector index...")

        chunks = self.processor.process_documents()
        embeddings = self.embedder.embed_documents(chunks)

        dimension = len(embeddings[0])
        self.vector_store = VectorStore(dimension)
        self.vector_store.add_embeddings(embeddings, chunks)
        self.vector_store.save()

    def _load_index(self):
        print("Loading existing vector index...")

        dummy_embedding = self.embedder.embed_text("dimension check")
        dimension = len(dummy_embedding)

        self.vector_store = VectorStore(dimension)
        self.vector_store.load()
