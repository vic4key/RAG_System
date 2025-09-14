from typing import List
from abc import ABC, abstractmethod

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Sinh embedding cho một đoạn văn bản."""
        pass

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model: str):
        import openai
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def embed(self, text: str) -> List[float]:
        import openai
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        return response['data'][0]['embedding']

class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
        """
        Azure OpenAI embedding provider using openai.AzureOpenAI client (OpenAI v1 SDK).
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint (e.g. https://xxx.openai.azure.com/)
            api_version: API version string (e.g. '2023-05-15')
            model: Deployment name (not model family)
        """
        import openai
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model
        self.client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, text: str) -> List[float]:
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        return embeddings

class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, endpoint: str, model: str):
        """
        Google AI Studio embedding provider using OpenAI-compatible API (OpenAI v1 SDK).
        Args:
            api_key: API key for Google AI Studio
            endpoint: Endpoint for Google AI Studio
            model: Model name (e.g. 'models/embedding-gecko-001')
        """
        import openai
        self.model = model
        self.client = openai.OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

def create_embedding_provider(provider: str, endpoint: str, model_name: str, api_key: str, api_version: str = None):
    """Factory function to create embedding provider by name"""
    provider = (provider if isinstance(provider, str) else "openai").strip().lower()

    if not all([provider, model_name]):
        raise Exception("Missing one or more required configuration parameters.")

    if provider == "azure":
        return AzureOpenAIEmbeddingProvider(api_key=api_key, endpoint=endpoint, api_version=api_version, model=model_name)
    elif provider == "openai":
        return OpenAIEmbeddingProvider(api_key=api_key, model=model_name)
    elif provider == "huggingface":
        return HuggingFaceEmbeddingProvider(model_name=model_name)
    elif provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(model_name=model_name)
    elif provider == "google":
        return GoogleEmbeddingProvider(api_key=api_key, endpoint=endpoint, model=model_name)
    else:
        raise ValueError(f"Unknown LLM_EMBEDDING_PROVIDER: {provider}")
