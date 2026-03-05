import os
import re
from typing import Dict, List

import fitz  # PyMuPDF

from app.config import parameters


class DocumentProcessor:

    def __init__(self):
        self.data_path = parameters.DATA_PATH
        self.chunk_size = parameters.CHUNK_SIZE
        self.chunk_overlap = parameters.CHUNK_OVERLAP

    # =============================
    # PUBLIC API
    # =============================

    def process_documents(self) -> List[Dict]:
        """
        Main pipeline:
        - Load documents
        - Extract text
        - Clean text
        - Chunk text
        - Attach metadata
        """
        processed_chunks = []

        for filepath in self._get_document_files():
            filename = os.path.basename(filepath)

            raw_text = self._extract_text(filepath)
            clean_text = self._clean_text(raw_text)
            chunks = self._chunk_text(clean_text)

            metadata_chunks = self._build_metadata(filename, chunks)
            processed_chunks.extend(metadata_chunks)

        return processed_chunks

    # =============================
    # INTERNAL STEPS
    # =============================

    def _get_document_files(self) -> List[str]:
        """
        Retrieve all PDF files from data directory.
        """
        return [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".pdf")
        ]

    def _extract_text(self, filepath: str) -> str:
        """
        Extract text from PDF using PyMuPDF.
        """
        text = ""

        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text") or ""

        return text

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning to improve embedding quality.
        """
        text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
        text = text.strip()
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using sliding window approach.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _build_metadata(self, filename: str, chunks: List[str]) -> List[Dict]:
        """
        Attach metadata to each chunk.
        """
        metadata_chunks = []

        for i, chunk in enumerate(chunks):
            metadata_chunks.append(
                {
                    "text": chunk,
                    "document": filename,
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                }
            )

        return metadata_chunks
