import os, sys
import logging
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.models.data_models import RAG_DataItem
from core.adapters.json_adapter import RAG_JSONFileAdapter

class FAQ_Item(RAG_DataItem):
    """Data structure specifically for FAQ items"""
    question: str
    answer: str

    def model_post_init(self, __context):
        """Post-initialization processing for FAQ items"""
        # Create combined content for embedding
        self.content = f"Câu hỏi: {self.question}\nTrả lời: {self.answer}"

        # Update metadata with question and answer
        self.metadata.update({
            "question": self.question,
            "answer": self.answer,
            "name": "faq"
        })

class SampleFAQsAdapter(RAG_JSONFileAdapter):
    """Provider specifically for loading and processing the sample FAQs"""

    @property
    def name(self) -> str:
        return "sample_faq"

    def load(self, file_path: str) -> List[RAG_DataItem]:
        """Load and process sample FAQs

        Args:
            file_path: Path to the sample FAQs JSON file.

        Returns:
            List of FAQ_Item objects created from the sample FAQs
        """
        data  = super().load(file_path)
        items = data["faqs"]

        # Process the sample FAQs
        documents = []

        # Process each FAQ item
        for idx, item in enumerate(items):
            question = item.get('question', '').strip()
            answer = item.get('answer', '')

            if not question or not answer:
                logging.warning(f"Skipping FAQ item {idx} due to missing question or answer")
                continue

            # Create FAQ_Item
            doc = FAQ_Item(
                id=f"faq_{idx}",
                question=question,
                answer=answer,
            )
            documents.append(doc)

        logging.info(f"Loaded {len(documents)} sample FAQ items")
        return documents
