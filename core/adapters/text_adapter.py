import os
import logging
from typing import List, Optional

from ..models.data_models import RAG_DataItem
from .base import RAG_Adapter

class RAG_TextItem(RAG_DataItem):
    """Data structure specifically for plain text chunks"""
    chunk_index: int
    source_file: str  # Chỉ lưu tên file thay vì toàn bộ nội dung

    def model_post_init(self, __context):
        """Post-initialization processing for plain text items"""
        # Update metadata with all fields for retrieval later
        self.metadata.update({
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
            "name": "text_file"
        })

class RAG_PlainTextAdapter(RAG_Adapter):
    """Adapter for loading and processing plain text files with LangChain's RecursiveCharacterTextSplitter"""

    @property
    def name(self) -> str:
        return "text_file"

    def load(self,
                  file_path: str,
                  chunk_size: int = 1000,
                  chunk_overlap: int = 200,
                  separators: Optional[List[str]] = None,
                  ) -> List[RAG_DataItem]:
        """Load and process plain text file using LangChain's RecursiveCharacterTextSplitter

        Args:
            file_path: Path to the plain text file
            chunk_size: Maximum size of chunks to split text into
            chunk_overlap: Number of characters of overlap between chunks
            separators: Optional list of separators to use for splitting text
                        Default is None, which uses RecursiveCharacterTextSplitter defaults

        Returns:
            List of RAG_TextItem objects created from the text chunks
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise Exception("langchain not installed. Please install it with: pip install langchain")

        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []

        # Read text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return []

        # Initialize text splitter
        splitter_args = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "length_function": len,
        }

        # Add separators if provided
        if separators:
            splitter_args["separators"] = separators

        text_splitter = RecursiveCharacterTextSplitter(**splitter_args)

        # Split text into chunks
        chunks = text_splitter.split_text(text)
        logging.info(f"Split text into {len(chunks)} chunks")

        # Create RAG_TextItem for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            # Create a unique ID based on file name and chunk index
            file_name = os.path.basename(file_path)
            item_id = f"text_{file_name}_{i}"

            # Create RAG_TextItem
            doc = RAG_TextItem(
                id=item_id,
                content=chunk,
                chunk_index=i,
                source_file=file_name,
                metadata={
                    "file_name": file_name,
                    "file_path": file_path,
                }
            )
            documents.append(doc)

        logging.info(f"Created {len(documents)} RAG_TextItem objects")
        return documents