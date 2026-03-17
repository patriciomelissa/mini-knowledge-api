from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStore
from app.processing.document_processor import DocumentProcessor


def rebuild_index() -> None:
    """
    Rebuild the vector index from source documents.

    This function executes the full indexing pipeline:
        1. Process documents from the data directory.
        2. Extract and prepare text chunks with metadata.
        3. Generate embeddings for all text chunks.
        4. Create a new vector store with the appropriate dimension.
        5. Add embeddings and metadata to the vector store.
        6. Persist the index and metadata to disk.

    This operation overwrites any existing index and is typically used
    when documents are updated or added.

    Returns:
        None
    """

    processor = DocumentProcessor()
    documents = processor.process_documents()

    embedder = EmbeddingService()

    texts = [doc["text"] for doc in documents]

    embeddings = embedder.embed_documents(texts)

    vector_store = VectorStore(len(embeddings[0]))

    vector_store.add_embeddings(embeddings, documents)

    vector_store.save()

    print("Index rebuilt successfully")


if __name__ == "__main__":
    """
    Entry point for manual index rebuilding.

    Executes the rebuild_index function when the script is run directly.
    """
    rebuild_index()
