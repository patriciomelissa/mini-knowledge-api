import os
from typing import Any, Dict, List, Optional

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

        retrieval_results = self._retrieve(question)

        if not retrieval_results:
            return self._build_empty_response()

        context = self._build_context(retrieval_results)

        answer = self.llm.generate_answer(context, question)

        return self._build_success_response(answer, retrieval_results)

    def reindex(self):
        """
        Force rebuild of the index.
        """
        self._create_index()
        self.is_initialized = True

    # -----------------------------
    # INTERNAL STEPS
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

        documents = self.processor.process_documents()

        raw_texts = [doc["text"] for doc in documents]

        embeddings = self.embedder.embed_documents(raw_texts)

        dimension = len(embeddings[0])
        self.vector_store = VectorStore(dimension)
        self.vector_store.add_embeddings(embeddings, documents)
        self.vector_store.save()

    def _load_index(self):
        print("Loading existing vector index...")

        dummy_embedding = self.embedder.embed_text("dimension check")
        dimension = len(dummy_embedding)

        self.vector_store = VectorStore(dimension)
        self.vector_store.load()

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(query_embedding)

        filtered = [res for res in results if res["score"] <= parameters.MAX_DISTANCE]

        return filtered

    def _build_context(self, results) -> str:
        return "\n\n".join(res["text"] for res in results)

    def _build_empty_response(self) -> Dict[str, str]:
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
        }

    def _build_success_response(self, answer: str, results: List) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": [
                {
                    "document": res["document"],
                    "chunk_id": res["chunk_id"],
                    "score": res["score"],
                }
                for res in results
            ],
        }
