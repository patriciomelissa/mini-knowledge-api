from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranking service using a cross-encoder model.

    This class uses a CrossEncoder to assign relevance scores to
    retrieved documents given a query. It then sorts the documents
    based on these scores to improve ranking quality.

    Attributes:
        model (CrossEncoder): Cross-encoder model for relevance scoring.
    """

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, question, documents):
        # Create pairs of (query, document_text)
        pairs = [(question, doc["text"]) for doc in documents]
        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Assign scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort documents by score descending
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        return documents
