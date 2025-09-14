from typing import Dict, Any
from pydantic import BaseModel, Field

class RAG_DataItem(BaseModel):
    """Base data structure for items in the RAG system"""
    id: str
    content: str = Field(default_factory=str)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    name: str = "generic"
    
    def to_vector_db_format(self) -> Dict[str, Any]:
        """Convert to format suitable for vector DB storage"""
        return {
            "id": self.id,
            "document": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_vector_result(cls, result_item: Dict[str, Any]) -> 'RAG_DataItem':
        """Create a RAG_DataItem from a vector DB query result item"""
        return cls(
            id=result_item.get("id", ""),
            content=result_item.get("document", ""),
            metadata=result_item.get("metadata", {})
        )
