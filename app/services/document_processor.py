import os
import re
from typing import List

import fitz  # PyMuPDF

from app.config import parameters


class DocumentProcessor:

    def load_documents(self) -> List[str]:
        """Loads all PDF files from data folder and returns raw text list."""
        documents = []
        for filename in os.listdir(parameters.DATA_PATH):
            if filename.endswith(".pdf"):
                path = os.path.join(parameters.DATA_PATH, filename)
                text = self._extract_text_from_pdf(path)
                documents.append(text)
        return documents

    def _extract_text_from_pdf(self, path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlap."""
        chunk_size = parameters.CHUNK_SIZE
        overlap = parameters.CHUNK_OVERLAP

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def process_documents(self) -> List[str]:
        """Full processing pipeline."""
        raw_docs = self.load_documents()

        all_chunks = []
        for doc in raw_docs:
            cleaned = self.clean_text(doc)
            chunks = self.chunk_text(cleaned)
            all_chunks.extend(chunks)

        return all_chunks