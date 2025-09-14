import os
import json
import logging
from typing import List, Dict, Any

import sys
# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.models.data_models import RAG_DataItem
from core.adapters.json_adapter import RAG_JSONFileAdapter

class EnglishFAQ_Item(RAG_DataItem):
    """Data structure specifically for English FAQ items"""
    category: str
    title: str
    instruction: str

    def model_post_init(self, __context):
        """Post-initialization processing for English FAQ items"""
        # Create combined content for embedding
        # Bao gồm cả category để tăng ngữ cảnh cho embedding
        self.content = f"Category: {self.category}\nTitle: {self.title}\nInstruction: {self.instruction}"

        # Update metadata with all fields for retrieval later
        self.metadata.update({
            "category": self.category,
            "title": self.title,
            "instruction": self.instruction,
            "name": "english_faq"
        })

class EnglishFAQsAdapter(RAG_JSONFileAdapter):
    """Provider specifically for loading and processing English FAQs"""

    @property
    def name(self) -> str:
        return "english_faq"

    def load(self, file_path: str) -> List[RAG_DataItem]:
        """Load and process English FAQs

        Args:
            file_path: Path to the English FAQs JSON file.

        Returns:
            List of EnglishFAQ_Item objects created from the English FAQs
        """
        items = super().load(file_path)

        # Process the English FAQs
        documents = []

        # Process each English FAQ item
        for item in items:
            item_id = str(item.get('id', ''))  # Convert numeric ID to string
            category = item.get('category', '').strip()
            title = item.get('title', '').strip()
            instruction = item.get('instruction', '')

            # Create EnglishFAQ_Item
            doc = EnglishFAQ_Item(
                id=f"eng_faq_{item_id}",
                category=category,
                title=title,
                instruction=instruction,
            )
            documents.append(doc)

        logging.info(f"Loaded {len(documents)} English FAQ items")
        return documents
