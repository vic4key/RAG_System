import os
import json
import logging
import chromadb
from typing import List, Dict, Any, Optional, Union
from openai import AzureOpenAI

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.4  # Ngưỡng độ tương đồng, giá trị thấp hơn = tương đồng hơn
DEFAULT_COLLECTION_NAME = "default_collection"
AZURE_API_VERSION = "2024-07-01-preview"

class RAG_System:
    """
    Lớp quản lý hệ thống RAG (Retrieval-Augmented Generation)
    """

    def __init__(self):
        """Khởi tạo hệ thống RAG với các thông số cấu hình cần thiết"""
        # Chỉ giữ lại các biến thực sự cần thiết cho hoạt động
        self._embedding_model_name = os.getenv("LLM_EMBEDDING_MODEL_NAME")
        self._collection_name = None

        # Khởi tạo embedding client
        self._embedding_client = AzureOpenAI(
            azure_endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_EMBEDDING_API_KEY_EMBEDDING"),
            api_version=AZURE_API_VERSION,
        )

        # Vector DB collection
        self._collection = None

    def _get_embedding(self, text: str) -> List[float]:
        """Tạo embedding từ văn bản sử dụng mô hình embedding đa ngôn ngữ."""
        try:
            response = self._embedding_client.embeddings.create(
                input=text,
                model=self._embedding_model_name
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Lỗi khi tạo embedding: {e}")
            return []

    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa văn bản: chuyển thành chữ thường và loại bỏ khoảng trắng thừa."""
        return text.lower().strip()

    def _load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Tải dữ liệu từ file JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('faqs', [])
        except Exception as e:
            logging.error(f"Lỗi khi đọc file dữ liệu: {e}")
            return []

    def _prepare_data(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chuẩn bị dữ liệu bao gồm chuẩn hóa và tạo trường nội dung kết hợp."""
        prepared_data = []
        for i, item in enumerate(items):
            question = self._normalize_text(item['question'])
            answer = item['answer']
            # Tạo nội dung kết hợp để tạo context phong phú
            combined_content = f"Câu hỏi: {question}\nTrả lời: {answer}"

            prepared_data.append({
                'id': str(i),
                'question': question,
                'answer': answer,
                'combined': combined_content,
                # Thêm metadata để phân loại và lọc
                'metadata': {}
            })
        return prepared_data

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

    def _add_data_to_vector_db(self, prepared_data: List[Dict[str, Any]]) -> None:
        """Thêm dữ liệu vào cơ sở dữ liệu vector."""
        if self._collection is None:
            logging.error("Vector DB chưa được khởi tạo")
            raise ValueError("Vector DB chưa được khởi tạo. Hãy tạo collection trước.")

        try:
            # Chuẩn bị dữ liệu cho việc thêm vào collection
            ids = [item['id'] for item in prepared_data]
            documents = [item['combined'] for item in prepared_data]
            metadatas = [
                {
                    'question': item['question'],
                    'answer': item['answer'],}
                for item in prepared_data
            ]

            # Tạo embeddings cho tất cả documents
            embeddings = []
            for doc in documents:
                embedding = self._get_embedding(doc)
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

    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """Truy vấn dữ liệu dựa trên câu hỏi của người dùng.

        Args:
            query_text: Câu hỏi của người dùng
            top_k: Số lượng kết quả trả về
            similarity_threshold: Ngưỡng độ tương đồng cho lọc kết quả.
                Nếu khác None, chỉ trả về best match nếu thỏa mãn ngưỡng, ngược lại trả về list rỗng.

        Returns:
            List[Dict[str, Any]]: Danh sách các kết quả phù hợp nhất

        Raises:
            ValueError: Nếu vector DB chưa được khởi tạo
        """
        if self._collection is None:
            logging.error("Vector DB chưa được khởi tạo")
            raise ValueError("Vector DB chưa được khởi tạo. Hãy gọi setup_from_file() trước.")

        try:
            # Chuẩn hóa truy vấn
            normalized_query = self._normalize_text(query_text)

            # Tạo embedding cho truy vấn
            query_embedding = self._get_embedding(normalized_query)

            # Đảm bảo có embedding hợp lệ
            if not query_embedding:
                logging.error("Không thể tạo embedding cho câu truy vấn")
                return []

            # Thực hiện truy vấn trên vector DB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )

            # Xử lý kết quả
            response_data = []
            if results and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    response_data.append({
                        'id': results['ids'][0][i],
                        'question': results['metadatas'][0][i]['question'],
                        'answer': results['metadatas'][0][i]['answer'],
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i],})

                # Lọc kết quả dựa trên ngưỡng tương đồng
                if similarity_threshold is not None:
                    if response_data and response_data[0]['distance'] <= similarity_threshold:
                        # Chỉ giữ lại best match nếu đạt ngưỡng
                        return [response_data[0]]
                    else:
                        # Trả về list rỗng nếu không đạt ngưỡng
                        return []

            return response_data
        except Exception as e:
            logging.error(f"Lỗi khi truy vấn dữ liệu: {e}")
            return []

    def setup_from_file(self, file_path: str, collection_name: str = DEFAULT_COLLECTION_NAME,
                       persist_directory: Optional[str] = None, recreate: bool = False) -> bool:
        """Thiết lập hệ thống RAG từ file JSON chứa dữ liệu.

        Args:
            file_path: Đường dẫn đến file JSON chứa dữ liệu
            collection_name: Tên của collection trong vector DB.
                Mặc định là "default_collection".
            persist_directory: Thư mục để lưu trữ vector DB.
                Nếu None, dữ liệu sẽ được lưu trong RAM.
            recreate: Nếu True, luôn tạo lại DB từ file.
                Nếu False và DB đã tồn tại, sẽ bỏ qua quá trình tạo lại.

        Returns:
            bool: True nếu thiết lập thành công

        Raises:
            ValueError: Nếu không thể tải dữ liệu từ file
            Exception: Nếu có lỗi khác xảy ra
        """
        try:
            # Cập nhật tên collection
            self._collection_name = collection_name

            # Tạo chroma client phù hợp với loại lưu trữ
            if persist_directory:
                logging.info(f"Sử dụng persistent storage tại: {persist_directory}")
                chroma_client = chromadb.PersistentClient(path=persist_directory)
            else:
                chroma_client = chromadb.Client()
                logging.info("Sử dụng in-memory storage")

            # Kiểm tra xem collection đã tồn tại và có dữ liệu chưa
            if not recreate:
                logging.info(f"Kiểm tra collection {collection_name} đã tồn tại chưa (recreate=False)")
                if self._check_collection_exists(chroma_client, collection_name):
                    # Collection đã tồn tại và có dữ liệu, không cần tạo lại
                    logging.info(f"Collection {collection_name} đã tồn tại và có dữ liệu, không cần tạo lại")
                    return True
                else:
                    logging.info(f"Collection {collection_name} không tồn tại hoặc không có dữ liệu, sẽ tạo mới")
            else:
                logging.info(f"recreate=True, sẽ tạo lại collection {collection_name}")

            # Đến đây, cần tạo mới hoặc tạo lại collection
            logging.info(f"Tạo collection mới{'(recreate=True)' if recreate else ''}")

            # Tải dữ liệu từ file JSON
            items = self._load_data(file_path)
            logging.info(f"Đã tải {len(items)} mục từ file")

            if not items:
                raise ValueError(f"Không thể tải dữ liệu từ file {file_path}")

            # Chuẩn bị dữ liệu
            prepared_data = self._prepare_data(items)

            # Xóa collection cũ nếu đã tồn tại
            try:
                chroma_client.delete_collection(name=collection_name)
                logging.info(f"Đã xóa collection cũ: {collection_name}")
            except Exception as e:
                # Bỏ qua lỗi nếu collection không tồn tại
                logging.debug(f"Collection chưa tồn tại hoặc không thể xóa: {e}")

            # Tạo collection mới
            self._collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Sử dụng cosine similarity
            )

            # Thêm dữ liệu vào vector DB
            self._add_data_to_vector_db(prepared_data)
            logging.info(f"Đã thêm {len(prepared_data)} mục vào vector DB")

            logging.info("Đã thiết lập hệ thống RAG thành công")
            return True
        except Exception as e:
            logging.error(f"Lỗi khi thiết lập hệ thống RAG: {e}")
            raise
