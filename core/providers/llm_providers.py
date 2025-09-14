from typing import Iterator, Optional
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, model: str, stream: bool = False, **kwargs):
        """Sinh response từ LLM cho một prompt."""
        pass

class OpenAI_LLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, endpoint: Optional[str] = None, api_version: Optional[str] = None):
        import openai
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.openai.com/v1"
        self.api_version = api_version
        openai.api_key = api_key
        openai.base_url = self.endpoint

    def generate(self, prompt: str, model: str, stream: bool = False, **kwargs):
        import openai
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        if stream:
            def stream_gen():
                for chunk in response:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            return stream_gen()
        else:
            return response.choices[0].message.content

class AzureOpenAI_LLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
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

    def generate(self, prompt: str, model: str, stream: bool, **kwargs):
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            **kwargs
        )
        if stream:
            def stream_gen():
                for chunk in response:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                    if delta:
                        yield delta
            return stream_gen()
        else:
            return response.choices[0].message.content

def create_llm_client(provider: str, endpoint: Optional[str], model_name: str, api_key: str, api_version: Optional[str] = None):
    provider = (provider or "openai").strip().lower()
    if provider == "azure":
        return AzureOpenAI_LLMProvider(api_key=api_key, endpoint=endpoint, api_version=api_version, model=model_name)
    elif provider == "openai":
        return OpenAI_LLMProvider(api_key=api_key, endpoint=endpoint, api_version=api_version)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
