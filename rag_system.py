import os
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Optional, Union

try:
    from core.adapters.base import RAG_Adapter
    from core.models.data_models import RAG_DataItem
    from core.utils.query_cache import QueryCache
    from core.providers.embedding_providers import create_embedding_provider
except:
    from .core.adapters.base import RAG_Adapter
    from .core.models.data_models import RAG_DataItem
    from .core.utils.query_cache import QueryCache
    from .core.providers.embedding_providers import create_embedding_provider

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.4  # Ngưỡng độ tương đồng, giá trị thấp hơn = tương đồng hơn
DEFAULT_COLLECTION_NAME = "default_collection"

class RAG_System:
    """
    Lớp quản lý hệ thống RAG (Retrieval-Augmented Generation)
    """

    def __init__(self, endpoint: str=None, model_name : str=None, api_key: str=None, api_version: str=None, provider: str=None):
        """Khởi tạo hệ thống RAG với các thông số cấu hình cần thiết"""
        self._endpoint = endpoint
        self._model_name = model_name
        self._api_key = api_key
        self._collection_name = DEFAULT_COLLECTION_NAME

        # Khởi tạo embedding client
        try:
            self._embedding_client = create_embedding_provider(
                provider=provider,
                endpoint=endpoint,
                model_name=model_name,
                api_key=api_key,
                api_version=api_version,
            )
        except Exception as e:
            logging.error(f"Error initializing embedding provider: {e}")
            self._embedding_client = None

        # Vector DB collection
        self._collection = None

        # Khởi tạo query cache dựa trên biến môi trường
        if bool(os.getenv("QUERY_CACHE_ENABLED") or '1'): # default is enabled
            self._query_cache = QueryCache()
        else:
            self._query_cache = None

    def _get_embedding(self, text: str) -> List[float]:
        """Tạo embedding từ văn bản sử dụng mô hình embedding."""
        if not self._embedding_client:
            logging.error("Embedding client not initialized")
            return []
        return self._embedding_client.embed(text=text)

    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa văn bản: chuyển thành chữ thường và loại bỏ khoảng trắng thừa."""
        return text.lower().strip()

    def _check_collection_exists(self, chroma_client: Union[chromadb.Client, chromadb.PersistentClient], collection_name: str) -> bool:
        """Kiểm tra xem collection đã tồn tại và có dữ liệu chưa.

        Args:
            chroma_client: ChromaDB client đã được khởi tạo (Client hoặc PersistentClient)
            collection_name: Tên của collection cần kiểm tra

        Returns:
            bool: True nếu collection tồn tại và có dữ liệu, False ngược lại
        """
        try:
            # Liệt kê tất cả collections
            collections = chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            logging.info(f"Danh sách collections hiện có: {collection_names}")

            # Thử kết nối với collection hiện có
            collection = chroma_client.get_collection(name=collection_name)

            # Kiểm tra xem collection có dữ liệu không bằng cách thử truy vấn
            peek = collection.peek(limit=1)
            if peek and len(peek['ids']) > 0:
                self._collection = collection
                logging.info(f"Đã kết nối thành công với collection hiện có: {collection_name}")
                return True
            else:
                logging.info(f"Collection {collection_name} tồn tại nhưng không có dữ liệu")
                return False
        except Exception as e:
            logging.info(f"Không tìm thấy collection {collection_name}: {e}")
            return False

    def _add_data_to_vector_db(self, prepared_data: List[RAG_DataItem]) -> None:
        """Thêm dữ liệu vào cơ sở dữ liệu vector."""
        if self._collection is None:
            logging.error("Vector DB chưa được khởi tạo")
            raise ValueError("Vector DB chưa được khởi tạo. Hãy tạo collection trước.")

        try:
            # Extract data from RAG_DataItem objects
            ids = [item.id for item in prepared_data]
            documents = [item.content for item in prepared_data]
            metadatas = [item.metadata for item in prepared_data]

            # Tạo embeddings cho tất cả documents
            embeddings = []
            for doc in documents:
                embedding = self._get_embedding(doc)
                if not embedding:
                    raise ValueError(f"Failed to create embedding for document: {doc[:50]}...")
                embeddings.append(embedding)

            # Thêm dữ liệu vào collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logging.info(f"Đã thêm {len(ids)} mục vào vector DB")
        except Exception as e:
            logging.error(f"Lỗi khi thêm dữ liệu vào vector DB: {e}")
            raise

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[RAG_DataItem]:
        """Truy vấn dữ liệu dựa trên câu hỏi của người dùng.

        Args:
            query_text: Câu hỏi của người dùng
            top_k: Số lượng kết quả trả về tối đa
            similarity_threshold: Ngưỡng độ tương đồng cho lọc kết quả

        Returns:
            List[RAG_DataItem]: Danh sách các kết quả phù hợp nhất
        """
        # Kiểm tra và sử dụng cache nếu được bật
        if self._query_cache:
            query_params = {
                "query_text": query_text,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "collection_name": self._collection_name,
            }

            if cache_result := self._query_cache.get(query_params):
                logging.info(f"Trả về kết quả từ cache cho truy vấn: {query_text[:50]}...")
                return cache_result
        else:
            query_params = {}

        if self._collection is None:
            logging.error("Vector DB chưa được khởi tạo")
            raise ValueError("Vector DB chưa được khởi tạo. Hãy thiết lập hệ thống trước.")

        try:
            # Chuẩn hóa và tạo embedding cho câu hỏi
            normalized_query = self._normalize_text(query_text)
            query_embedding = self._get_embedding(normalized_query)

            if not query_embedding:
                raise ValueError("Failed to create embedding for query")

            # Truy vấn vector DB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Kiểm tra ngưỡng độ tương đồng nếu có yêu cầu
            response_items = []

            if results and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Nếu có ngưỡng và khoảng cách lớn hơn ngưỡng, bỏ qua
                    if similarity_threshold is not None and results['distances'][0][i] > similarity_threshold:
                        continue

                    # Tạo RAG_DataItem từ kết quả
                    item = RAG_DataItem(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        metadata={
                            **results['metadatas'][0][i],
                            'distance': results['distances'][0][i]
                        }
                    )
                    response_items.append(item)

            # Lưu kết quả vào cache nếu được bật
            if self._query_cache and query_params and response_items:
                self._query_cache.set(query_params, response_items)

            return response_items
        except Exception as e:
            logging.error(f"Lỗi khi truy vấn: {e}")
            raise

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
        if not self._embedding_client:
            raise Exception("Embedding client is not initialized")

        try:
            self._collection_name = collection_name

            # Khởi tạo ChromaDB client
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                chroma_client = chromadb.PersistentClient(path=persist_directory)
                logging.info(f"Khởi tạo ChromaDB với lưu trữ tại {persist_directory}")
            else:
                chroma_client = chromadb.Client(Settings())
                logging.info("Khởi tạo ChromaDB trong bộ nhớ")

            # Kiểm tra xem collection đã tồn tại và có dữ liệu chưa
            if self._check_collection_exists(chroma_client, collection_name) and not recreate:
                logging.info(f"Collection {collection_name} đã tồn tại và có dữ liệu, sử dụng lại")
                self._collection = chroma_client.get_collection(name=collection_name)
                return True

            # Xóa collection cũ nếu cần tạo lại
            if recreate:
                try:
                    chroma_client.delete_collection(name=collection_name)
                    logging.info(f"Đã xóa collection cũ: {collection_name}")
                except Exception as e:
                    logging.debug(f"Collection chưa tồn tại hoặc không thể xóa: {e}")

            # Tải dữ liệu từ adapter
            logging.info(f"Đang tải dữ liệu từ adapter {adapter.__class__.__name__}")
            data_items = adapter.load(**adapter_params)

            if not data_items:
                logging.warning("Không có dữ liệu nào được tải")
                return False

            # Tạo collection mới
            self._collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Sử dụng cosine similarity
            )

            # Thêm dữ liệu vào vector DB
            self._add_data_to_vector_db(data_items)

            logging.info(f"Đã thiết lập thành công hệ thống RAG với {len(data_items)} mục dữ liệu")

            # Xóa cache nếu đang tạo lại collection hoặc đổi collection
            if self._query_cache:
                self._query_cache.clear()
                logging.info("Đã xóa cache truy vấn")

            return True

        except Exception as e:
            logging.error(f"Lỗi khi thiết lập hệ thống RAG: {e}")
            raise
