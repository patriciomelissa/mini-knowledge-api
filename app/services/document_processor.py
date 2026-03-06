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
        documents = []

        for filepath in self.get_files():

            pages = self.extract_pages(filepath)
            for page_number, text in pages:
                cleaned = self.clean_text(text)
                chunks = self.chunk_text(cleaned)
                docs = self.build_metadata(
                    os.path.basename(filepath), chunks, page_number
                )

                documents.extend(docs)

        return documents

    # =============================
    # INTERNAL STEPS
    # =============================

    def get_files(self) -> List[str]:
        """
        Retrieve all PDF files from data directory.
        """
        return [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".pdf")
        ]

    def extract_text(self, filepath: str) -> str:
        """
        Extract text from PDF using PyMuPDF.
        """
        text = ""

        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text") or ""

        return text

    def extract_pages(self, filepath: str) -> list:
        """ """
        pages = []

        with fitz.open(filepath) as doc:
            for page_number, page in enumerate(doc):
                text = page.get_text("text")
                pages.append((page_number + 1, text))

        return pages

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning to improve embedding quality.
        """
        text = re.sub(r"\s+", " ", text)  # Remove excessive whitespace
        text = text.strip()
        return text

    def chunk_text(self, text: str) -> List[str]:
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

    def build_metadata(
        self, filename: str, chunks: List[str], page_number: int
    ) -> List[Dict]:
        """
        Attach metadata to each chunk.
        """
        metadata_chunks = []

        for i, chunk in enumerate(chunks):
            metadata_chunks.append(
                {
                    "text": chunk,
                    "document": filename,
                    "page": page_number,
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                }
            )

        return metadata_chunks
