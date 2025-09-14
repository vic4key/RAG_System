import os
import logging
from typing import List, Optional

from RAG_System.core.config import EmbConfig, LLMConfig
from RAG_System.core.adapters.base import RAG_Adapter
from RAG_System.core.models.data_models import RAG_DataItem
from RAG_System.core.rag_generation import RAG_Generation
from RAG_System.core.rag_retrieval import RAG_Retrieval, DEFAULT_TOP_K, SIMILARITY_THRESHOLD, DEFAULT_COLLECTION_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAG_System():
    """
    Lớp quản lý hệ thống RAG (Retrieval-Augmented Generation)
    """
    def __init__(self, emb_config: EmbConfig = None, llm_config: LLMConfig = None):
        """
        Initializes the RAG system with embedding and language model configurations.
        Args:
            emb_config (EmbConfig): Configuration for the embedding provider.
            llm_config (LLMConfig): Configuration for the llm provider.
        """
        get_var = lambda attr, env: getattr(emb_config, attr, None) if emb_config else os.getenv(env)

        try:
            self._retrieval = RAG_Retrieval(
                emb_provider=get_var("provider", "LLM_EMBEDDING_PROVIDER"),
                emb_endpoint=get_var("endpoint", "LLM_EMBEDDING_ENDPOINT"),
                emb_model_name=get_var("model_name", "LLM_EMBEDDING_MODEL_NAME"),
                emb_api_key=get_var("api_key", "LLM_EMBEDDING_API_KEY"),
                emb_api_version=get_var("api_version", "LLM_EMBEDDING_API_VERSION"),
            )
        except:
            self._retrieval = None

        try:
            self._generation = RAG_Generation(
                llm_provider=get_var("provider", "LLM_PROVIDER"),
                llm_endpoint=get_var("endpoint", "LLM_ENDPOINT"),
                llm_model_name=get_var("model_name", "LLM_MODEL_NAME"),
                llm_api_key=get_var("api_key", "LLM_API_KEY"),
                llm_api_version=get_var("api_version", "LLM_API_VERSION"),
            )
        except:
            self._generation = None

    def setup_from_adapter(self,
                            adapter: RAG_Adapter,
                            adapter_params: dict,
                            collection_name: str = DEFAULT_COLLECTION_NAME,
                            persist_directory: Optional[str] = None,
                            recreate: bool = False) -> bool:
        """Thiết lập hệ thống RAG với adapter dữ liệu.

        Args:
            adapter: Adapter để xử lý riêng cho từng loại dữ liệu
            adapter_params: Các tham số sử dụng cho adapter
            collection_name: Tên của collection trong vector DB
            persist_directory: Thư mục để lưu trữ vector DB
            recreate: Nếu True, luôn tạo lại DB

        Returns:
            bool: True nếu thiết lập thành công
        """
        if not self._retrieval:
            raise Exception("RAG Retrieval not initialized")

        return self._retrieval.setup_from_adapter(
            adapter=adapter,
            adapter_params=adapter_params,
            collection_name=collection_name,
            persist_directory=persist_directory,
            recreate=recreate,
        )

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[RAG_DataItem]:
        """Truy vấn dữ liệu dựa trên câu hỏi của người dùng.

        Args:
            query_text: Câu hỏi của người dùng
            top_k: Số lượng kết quả trả về tối đa
            similarity_threshold: Ngưỡng độ tương đồng cho lọc kết quả

        Returns:
            List[RAG_DataItem]: Danh sách các kết quả phù hợp nhất
        """
        if not self._retrieval:
            raise Exception("RAG Retrieval not initialized")

        return self._retrieval.query(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def generate(self, query: str, retrieved_items: List[RAG_DataItem], stream: bool = False, **llm_kwargs):
        if not self._generation:
            raise Exception("RAG Generation not initialized")

        return self._generation.generate(
            query=query,
            retrieved_items=retrieved_items,
            stream=stream,
            **llm_kwargs,
        )
