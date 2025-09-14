from typing import List
from abc import ABC, abstractmethod

from ..models.data_models import RAG_DataItem

class RAG_Adapter(ABC):
    """Abstract base class for all data adapters"""

    @abstractmethod
    def load(self, **kwargs) -> List[RAG_DataItem]:
        """Load data from source and convert to RAG_DataItem format

        Args:
            **kwargs: Additional parameters specific to each adapter

        Returns:
            List of RAG_DataItem objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this adapter"""
        pass