import os
import time
from typing import Any, Dict, List, Optional

from app.config import parameters
from app.core.embeddings import EmbeddingService
from app.core.llm_local import LocalLLMService
from app.core.reranker import Reranker
from app.core.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor
from observability.logging_config import setup_logging
from observability.rag_tracer import RAGTracer

setup_logging()


class RAGService:
    """
    Retrieval-Augmented Generation (RAG) service pipeline.

    This service orchestrates document retrieval and language model generation
    using embeddings and vector similarity search.

    Pipeline overview:
        1. Ensure system initialization.
        2. Retrieve relevant document chunks for a given question.
        3. If no relevant documents are found, return an empty response.
        4. Construct context from retrieved results.
        5. Generate an answer using an LLM based on context.
        6. Return the answer along with retrieval metadata.

    Attributes:
        embedder (EmbeddingService): Service used to generate embeddings.
        processor (DocumentProcessor): Service used to process source documents.
        vector_store (Optional[VectorStore]): Vector database instance.
        is_initialized (bool): Initialization state flag.
        llm (LocalLLMService): Language model service used for answer generation.
    """

    def __init__(self) -> None:
        self.embedder = EmbeddingService()
        self.processor = DocumentProcessor()
        self.vector_store: Optional[VectorStore] = None
        self.reranker = Reranker()
        self.is_initialized = False
        self.llm = LocalLLMService()
        self.tracer = RAGTracer()

    # -----------------------------
    # PUBLIC API
    # -----------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Process a user question using a retrieval-augmented generation pipeline.

        Steps:
        1. Ensure the system is initialized. Log the incoming user query.
        2. Retrieve relevant documents or chunks related to the question.
        Log retrieval statistics (number of chunks, similarity score, pages).
        3. If no relevant information is found, return an empty response.
        4. Apply cross-encoder reranking to improve document ranking quality.
        5. Build a context from the reranked results while respecting context
        size limits. Log context statistics.
        6. Generate an answer using the LLM based on the context and question.
        Log LLM response time.
        7. Return the generated answer along with retrieval metadata.

        Args:
            question (str): User question to be answered.

        Returns:
            Dict[str, Any]: Response containing:
            - answer (str): Generated answer from the LLM.
            - sources (List[Dict[str, Any]]): Retrieved document metadata used
             as context.
        """
        self.ensure_initialized()

        # logging
        self.tracer.trace_query(question)

        retrieval_results = self.retrieve(question)

        # logging
        self.tracer.trace_retrieval(retrieval_results)

        if not retrieval_results:
            return self.build_empty_response()

        # apply reranking
        reranked_results = self.reranker.rerank(question, retrieval_results)

        context = self.build_context_using_charslimit(reranked_results)

        # logging
        self.tracer.trace_context(context)
        start = time.time()

        answer = self.llm.generate_answer(context, question)

        # logging
        self.tracer.trace_llm(start)

        return self.build_success_response(answer, reranked_results)

    def reindex(self) -> None:
        """
        Force rebuild of the vector index.

        This method recreates the embedding index from scratch.
        """
        self.create_index()
        self.is_initialized = True

    # -----------------------------
    # INTERNAL STEPS
    # -----------------------------

    def initialize(self) -> None:
        """
        Safely initialize the RAG service.

        This method can be called multiple times without side effects.
        """
        if self.is_initialized:
            return

        if self.index_exists():
            self.load_index()
        else:
            self.create_index()

        self.is_initialized = True

    def ensure_initialized(self):
        """
        Lazy initialization guard.

        Ensures the vector index is loaded or created before use.
        """
        if not self.is_initialized:
            self.initialize()

    def index_exists(self) -> bool:
        """
        Check if vector index storage file exists.

        Returns:
            bool: True if index file exists, False otherwise.
        """
        index_path = os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss")
        return os.path.exists(index_path)

    def create_index(self) -> None:
        """
        Create a new vector index from processed documents.
        """
        print("Creating new vector index...")

        documents = self.processor.process_documents()

        raw_texts = [doc["text"] for doc in documents]

        embeddings = self.embedder.embed_documents(raw_texts)

        dimension = len(embeddings[0])
        self.vector_store = VectorStore(dimension)
        self.vector_store.add_embeddings(embeddings, documents)
        self.vector_store.save()

    def load_index(self) -> None:
        """
        Load an existing vector index from storage.

        """
        print("Loading existing vector index...")

        self.vector_store = VectorStore(0)
        self.vector_store.load()

        # print("Index size:", self.vector_store.index.ntotal)
        # print("Metadata size:", len(self.vector_store.metadata))

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on question embedding similarity.

        Args:
            question (str): User question.

        Returns:
            List[Dict[str, Any]]: Retrieval search results filtered by distance
            threshold.
        """
        query_embedding = self.embedder.embed_text(question)
        results = self.vector_store.search(query_embedding)

        # for now, this filter is doing inside of the search() funtion
        # filtered = [res for res in results if res["score"] > parameters.MIN_SCORE]

        return results

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieval results.

        Args:
            results (List[Dict[str, Any]]): Retrieval results.

        Returns:
            str: Concatenated context text.
        """
        return "\n\n".join(res["text"] for res in results)

    def build_context_with_metadata(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string with metadata from retrieval results.

        Args:
            results (List[Dict[str, Any]]): Retrieval results.

        Returns:
            str: Concatenated context text with metadata.
        """
        context_parts = []

        for r in results:
            context_parts.append(
                f"[Document: {r['document']} | Page: {r['page']}]\n{r['text']}"
            )

        return "\n\n".join(context_parts)

    def build_context_using_charslimit(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string with metadata from retrieval results
        but, limit context using characters limit.

        Args:
            results (List[Dict[str, Any]]): Retrieval results.

        Returns:
            str: Concatenated context text with metadata.
        """
        context_parts = []
        counter_size = 0

        for r in results:
            text = r["text"]

            if counter_size + len(text) > parameters.MAX_CONTEXT_CHARS:
                break

            context_parts.append(text)
            counter_size += len(text)

        return "\n\n".join(context_parts)

    def build_empty_response(self) -> List[Dict[str, Any]]:
        """
        Build response structure when no retrieval results are found.

        Returns:
            Dict[str, Any]: Empty response template.
        """
        return {
            "answer": "I could not find relevant information in the documents.",
            "sources": [],
        }

    def build_success_response(
        self, answer: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build successful response structure.

        Args:
            answer (str): Generated answer.
            results (List[Dict[str, Any]]): Retrieval results.

        Returns:
            Dict[str, Any]: Response containing answer and source metadata.
        """
        return {
            "answer": answer,
            "sources": [
                {
                    "document": res["document"],
                    "page": res["page"],
                    "text": res["text"],
                    "score": res["score"],
                }
                for res in results
            ],
        }
