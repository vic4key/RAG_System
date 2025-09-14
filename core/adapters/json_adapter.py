import json
import os
import logging
from typing import List

from ..models.data_models import RAG_DataItem
from .base import RAG_Adapter

class RAG_JSONFileAdapter(RAG_Adapter):
    """Generic adapter for loading data from JSON files"""

    @property
    def name(self) -> str:
        return "json_file"

    def load(self, file_path: str) -> List[RAG_DataItem]:
        """Load raw data from a JSON file and process it

        Args:
            file_path: Path to the JSON file
            **kwargs: Additional parameters specific to JSON processing

        Returns:
            List of RAG_DataItem objects created from the JSON data
        """
        if not file_path:
            raise ValueError("file_path must be provided for RAG_JSONFileAdapter")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        result = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                result = json.load(file)
            logging.info(f"Loaded JSON data from {file_path}")

        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            raise
        return result