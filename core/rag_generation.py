import logging
from typing import List

from RAG_System.core.models.data_models import RAG_DataItem
from RAG_System.core.providers.llm_providers import create_llm_client

class RAG_Generation:
    """
    RAG (Generation)
    """

    def __init__(self, llm_provider: str, llm_endpoint: str, llm_model_name: str, llm_api_key: str, llm_api_version: str = None):
        """
        Initializes the llm client for the generation system.

        Args:
            llm_provider (str): The name of the LLM provider (e.g., 'openai', 'azure').
            llm_endpoint (str): The endpoint URL for the LLM service.
            llm_model_name (str): The name or identifier of the LLM model to use.
            llm_api_key (str): The API key for authenticating with the LLM provider.
            llm_api_version (str, optional): The version of the LLM API to use. Defaults to None.

        Raises:
            Logs an error and sets the LLM client to None if initialization fails.
        """
        try:
            self._llm_model_name = llm_model_name
            self._llm_client = create_llm_client(
                provider=llm_provider,
                endpoint=llm_endpoint,
                model_name=llm_model_name,
                api_key=llm_api_key,
                api_version=llm_api_version,
            )
        except Exception as e:
            logging.error(f"Error initializing LLM provider: {e}")
            self._llm_client = None

    def _build_prompt(self, query: str, retrieved_items: List[RAG_DataItem]) -> str:
        # TODO: Consider formatting the context for clarity and including metadata such as file names or sources.
        context = "\n\n".join([
            f"[Source: {item.metadata.get('file_name', item.metadata.get('name', ''))}]\n{item.content.strip()}"
            for item in retrieved_items
        ])
        prompt = f"""Dưới đây là các đoạn thông tin liên quan:\n{context}\n---\nCâu hỏi: {query}\nTrả lời:"""
        return prompt

    def generate(self, query: str, retrieved_items: List[RAG_DataItem], stream: bool = False, **llm_kwargs):
        if not self._llm_client:
            raise Exception("LLM client is not initialized")

        if not retrieved_items:
            return "Không tìm thấy kết quả"

        prompt = self._build_prompt(query, retrieved_items)

        response = self._llm_client.generate(prompt, model=self._llm_model_name, stream=stream, **llm_kwargs)

        if stream:
            def stream_gen():
                for chunk in response:
                    yield chunk
            return stream_gen()
        else:
            # Full text response
            return response
