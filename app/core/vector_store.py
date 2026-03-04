import os
from typing import List, Tuple

import faiss
import numpy as np

from app.config import parameters


class VectorStore:

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = None) -> List[Tuple[str, float]]:
        if top_k is None:
            top_k = parameters.TOP_K

        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], float(dist)))

        return results

    def save(self):
        os.makedirs(parameters.VECTOR_STORE_PATH, exist_ok=True)
        faiss.write_index(self.index, os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss"))

        with open(os.path.join(parameters.VECTOR_STORE_PATH, "chunks.txt"), "w", encoding="utf-8") as f:
            for chunk in self.text_chunks:
                f.write(chunk.replace("\n", " ") + "\n")

    def load(self):
        index_path = os.path.join(parameters.VECTOR_STORE_PATH, "index.faiss")
        chunks_path = os.path.join(parameters.VECTOR_STORE_PATH, "chunks.txt")

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)

        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.text_chunks = f.read().splitlines()