import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger("rag")


class RAGTracer:
    """
    Tracing and logging utility for the RAG pipeline.

    This class provides structured logging for key stages of the
    Retrieval-Augmented Generation workflow, helping monitor system
    behavior and performance.

    Traced stages include:
        - User queries
        - Retrieval results
        - Context size passed to the LLM
        - LLM response time

    The tracer relies on an external logger instance to record events.
    """

    def trace_query(self, question: str) -> None:
        """
        Log the incoming user query.

        Args:
            question (str): User question submitted to the RAG system.
        """
        logger.info(f"[QUERY] {question}")

    def trace_retrieval(self, results: List[Dict[str, Any]]) -> None:
        """
        Log retrieval statistics from the vector search.

        Args:
            results (List[Dict[str, Any]]): Retrieved document chunks
            returned by the vector store.
        """
        if not results:
            logger.warning("[RETRIEVAL] no results found")
            return

        best_score = results[0]["score"]
        num_chunks = len(results)
        pages = [r["page"] for r in results]

        logger.info(
            f"[RETRIEVAL] chunks={num_chunks} best_score={best_score:.3f} pages={pages}"
        )

    def trace_context(self, context: str) -> None:
        """
        Log context statistics before sending it to the LLM.

        Args:
            context (str): Context string constructed from retrieved documents.
        """

        logger.info(f"[CONTEXT] size_chars={len(context)}")

    def trace_llm(self, start_time: float) -> None:
        """
        Log LLM response latency.

        Args:
            start_time (float): Timestamp recorded before the LLM call.
        """

        duration = time.time() - start_time

        logger.info(f"[LLM] response_time={duration:.2f}s")
