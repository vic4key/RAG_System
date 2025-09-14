from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbConfig:
    provider: Optional[str] = None
    endpoint: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None

@dataclass
class LLMConfig:
    provider: Optional[str] = None
    endpoint: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
