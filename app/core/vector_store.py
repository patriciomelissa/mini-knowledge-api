import json
import os
from typing import Any, Dict, List

import faiss
import numpy as np

from app.config import parameters


class VectorStore:
    """
    Vector similarity search storage using FAISS index.

    This class manages embedding storage, similarity search operations,
    and persistence of vector index and document metadata.

    The store supports:
        - Adding embeddings and associated document metadata.
        - Performing nearest neighbor search.
        - Saving and loading index and metadata from disk.

    Attributes:
        dimension (int): Embedding vector dimension.
        index (faiss.IndexFlatIP): FAISS index.
        metadata (List[Dict[str, Any]]): Document metadata storage.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_embeddings(
        self, embeddings: List[List[float]], documents: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings and corresponding document metadata to the vector store.

        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            documents (List[Dict[str, Any]]): List of document metadata dictionaries.
        """
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(documents)

    def search(
        self, query_embedding: List[float], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector index.

        Args:
            query_embedding (List[float]): Query embedding vector.
            top_k (Optional[int]): Number of top results to retrieve.

        Returns:
            List[Dict[str, Any]]: Ranked search results.
        """
        if top_k is None:
            top_k = parameters.TOP_K

        query_vector = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # filter results using MIN_SCORE parameters
            # improve the quality of the results
            # we want to keep the values > MIN_SCORE
            if score < parameters.MIN_SCORE:
                continue

            if idx < len(self.metadata):
                results.append(
                    {
                        "text": self.metadata[idx]["text"],
                        "document": self.metadata[idx]["document"],
                        "page": self.metadata[idx]["page"],
                        "chunk_id": self.metadata[idx]["chunk_id"],
                        "score": float(score),
                    }
                )

        return results

    def save(self) -> None:
        """
        Persist vector index and metadata to disk.
        """
        # create the vector_store folder
        os.makedirs(parameters.VECTOR_STORE_PATH, exist_ok=True)
        # create the index.faiss file
        faiss.write_index(
            self.index, os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss")
        )

        # create the json file with metadata
        with open(
            os.path.join(parameters.VECTOR_STORE_PATH, "chunks_metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=4)

    def load(self) -> None:
        """
        Load vector index and metadata from disk if they exist.
        """
        index_path = os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss")
        chunks_metadata_path = os.path.join(
            parameters.VECTOR_STORE_PATH, "chunks_metadata.json"
        )

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

        if os.path.exists(chunks_metadata_path):
            with open(chunks_metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Vector index and metadata are inconsistent")
