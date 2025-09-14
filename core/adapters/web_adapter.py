import logging
import requests
from typing import List, Dict, Any

from .base import RAG_Adapter
from ..models.data_models import RAG_DataItem

class RAG_WebAPIAdapter(RAG_Adapter):
    """Adapter for loading data from web APIs"""

    @property
    def name(self) -> str:
        return "web_api"

    def load(self, endpoint: str, api_key: str = None, headers: Dict[str, str] = None) -> List[RAG_DataItem]:
        """Load data from a web API

        Args:
            api_key: Optional API key for authentication
            headers: Optional additional headers
            **kwargs: Additional parameters for API requests

        Returns:
            List of RAG_DataItem objects created from the API response
        """
        if not endpoint:
            raise ValueError("endpoint must be provided for RAG_WebAPIAdapter")

        # Prepare headers
        request_headers = headers or {}
        if api_key:
            request_headers['Authorization'] = f'Bearer {api_key}'

        try:
            response = requests.get(endpoint, headers=request_headers)
            response.raise_for_status()

            data = response.json()
            logging.info(f"Loaded data from API '{endpoint}'")

            return self._process_data(data)

        except Exception as e:
            logging.error(f"Error loading data from API '{endpoint}': {str(e)}")
            raise

    def _process_data(self, data: Dict[str, Any], **kwargs) -> List[RAG_DataItem]:
        """Process the API response data into RAG_DataItem objects

        This method should be overridden by subclasses to handle specific API response formats.

        Args:
            endpoint: The loaded API data
            **kwargs: Additional parameters for processing

        Returns:
            List of RAG_DataItem objects
        """
        # Override this method to process the loaded data into RAG_DataItem objects
        return []