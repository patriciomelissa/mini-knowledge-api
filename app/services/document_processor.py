import os
import re
from typing import Dict, List

import fitz  # PyMuPDF

from app.config import parameters


class DocumentProcessor:
    """
    Document processing pipeline for extracting, cleaning, and chunking PDF text.

    This class is responsible for preparing documents for embedding generation by:
        - Loading PDF files from the data directory.
        - Extracting text from PDF pages.
        - Cleaning raw text.
        - Splitting text into manageable chunks.
        - Attaching metadata to chunks.

    Attributes:
        data_path (str): Directory path containing source documents.
        chunk_size (int): Size of text chunks.
        chunk_overlap (int): Overlap size between consecutive chunks.
    """

    def __init__(self) -> None:
        self.data_path = parameters.DATA_PATH
        self.chunk_size = parameters.CHUNK_SIZE
        self.chunk_overlap = parameters.CHUNK_OVERLAP
        self.min_chunk_size = parameters.MIN_CHUNK_SIZE

    # =============================
    # PUBLIC API
    # =============================

    def process_documents(self) -> List[Dict]:
        """
        Main document processing pipeline.

        Steps:
            1. Load documents.
            2. Extract text from PDF pages.
            3. Clean extracted text.
            4. Split text int paragraphs
            5. Create chunks text.
            6. Attach metadata to chunks.

        Returns:
            List[Dict]: List of processed document chunk metadata.
        """
        documents = []

        for filepath in self.get_files():

            pages = self.extract_pages(filepath)
            for page_number, text in pages:
                cleaned = self.clean_text(text)
                # paragraph normalization
                paragraphs = self.split_paragraphs(cleaned)
                full_text = " ".join(paragraphs)
                chunks = self.create_chunks(full_text)
                chunks_list = self.enforce_min_chunk_size(chunks)
                docs = self.build_metadata(
                    os.path.basename(filepath), chunks_list, page_number
                )

                documents.extend(docs)

        return documents

    # =============================
    # INTERNAL STEPS
    # =============================

    def get_files(self) -> List[str]:
        """
        Retrieve all PDF files from the data directory.

        Returns:
            List[str]: List of PDF file paths.
        """
        return [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".pdf")
        ]

    def extract_text(self, filepath: str) -> str:
        """
        Extract full text from a PDF file.

        Args:
            filepath (str): Path to PDF file.

        Returns:
            str: Extracted text content.
        """
        text = ""

        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text") or ""

        return text

    def extract_pages(self, filepath: str) -> List:
        """
        Extract text page by page from PDF file.

        Args:
            filepath (str): Path to PDF file.

        Returns:
            List: List of tuples (page_number, page_text).
        """
        pages = []

        with fitz.open(filepath) as doc:
            for page_number, page in enumerate(doc):
                text = page.get_text("text")
                pages.append((page_number + 1, text))

        return pages

    def clean_text(self, text: str) -> str:
        """
        Clean raw PDF text before chunking.

        Args:
            text (str): Raw text.

        Returns:
            str: Cleaned text.
        """
        # 1. remove hyphen line breaks
        text = re.sub(r"-\n", "", text)

        # 2. replace line breaks with space
        text = re.sub(r"\n+", " ", text)

        # 3. remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # 4. remove page numbers alone
        text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)

        return text.strip()

    def split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs

        Parameters
        ----------
        text : str
            Raw text

        Returns
        -------
            List[str]: List of paragraphs.
        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if len(p.strip()) > 40]

    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using sliding window strategy.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of text chunks.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def enforce_min_chunk_size(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks smaller than self.min_chunk_size with adjacent chunks.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List[str]: List of text chunks where each chunk meets the minimum size.
        """
        if not chunks:
            return []

        merged_chunks = []
        buffer = ""

        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                # acumula no buffer
                if buffer:
                    buffer += " " + chunk
                else:
                    buffer = chunk
            else:
                # antes de adicionar o chunk grande, junta o buffer se existir
                if buffer:
                    merged_chunks.append(buffer.strip())
                    buffer = ""
                merged_chunks.append(chunk.strip())

        # no final, adiciona qualquer buffer restante
        if buffer:
            if merged_chunks:
                merged_chunks[-1] += " " + buffer
            else:
                merged_chunks.append(buffer.strip())

        return merged_chunks

    def build_metadata(
        self, filename: str, chunks: List[str], page_number: int
    ) -> List[Dict]:
        """
        Build metadata dictionary for each text chunk.

        Args:
            filename (str): Source document filename.
            chunks (List[str]): List of text chunks.
            page_number (int): Page number where chunk was extracted.

        Returns:
            List[Dict]: List of chunk metadata dictionaries.
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
