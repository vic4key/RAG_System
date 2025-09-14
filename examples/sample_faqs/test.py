import os
import logging
import unittest

from RAG_System import RAG_System
from RAG_System.core.config import EmbConfig
from sample_faqs_adapter import SampleFAQsAdapter

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.70

class TestRAGSystem(unittest.TestCase):
    """Test cases cho RAG System với cấu trúc mới"""

    @classmethod
    def setUpClass(cls):
        """Thiết lập hệ thống RAG một lần cho tất cả các test case"""
        print("\n=== THIẾT LẬP HỆ THỐNG RAG CHO TEST ===")

        # Thiết lập tên file và đường dẫn lưu trữ
        file_path = os.path.join(os.path.dirname(__file__), 'sample_faqs.json')
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = os.path.join(os.getcwd(), f"chroma_db/{file_name_without_ext}")

        emb_config = EmbConfig(
            provider=os.getenv("LLM_EMBEDDING_PROVIDER"),
            endpoint=os.getenv("LLM_EMBEDDING_ENDPOINT"),
            model_name=os.getenv("LLM_EMBEDDING_MODEL_NAME"),
            api_key=os.getenv("LLM_EMBEDDING_API_KEY"),
            api_version=os.getenv("LLM_EMBEDDING_API_VERSION"),
        )

        # Set up with the provider using persistent storage
        print(f"Setting up RAG with persistent_directory={persist_dir}, recreate=False")
        cls.rag_system = RAG_System(emb_config=emb_config)

        adapter = SampleFAQsAdapter()

        # Sử dụng persistent database để lưu trữ trên đĩa
        print(f"Thiết lập RAG với persistent_directory={persist_dir}, recreate=False")
        cls.rag_system.setup_from_adapter(
            adapter=adapter,
            adapter_params={
                "file_path": file_path,
            },
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,
            recreate=False,
        )

        print(f"Ngưỡng tương đồng (SIMILARITY_THRESHOLD): {TEST_SIMILARITY_THRESHOLD}")
        print("=== THIẾT LẬP HOÀN TẤT ===\n")

        # Kiểm thử với một vài câu hỏi
        cls.test_questions = [
            "Làm sao để chỉnh sửa hoặc xóa chi tiêu?",
            "xóa chi tiêu?",
            "How do I connect my bank account?",
            "Can I scan my receipts?",
            "How to scan a receipt?",
        ]

    def test_queries(self):
        """Test truy vấn có ngưỡng tương đồng"""

        print("\n=== TEST QUERIES WITH THRESHOLD ===")

        # Lưu kết quả để tạo summary
        summary_results = []

        for idx, question in enumerate(self.test_questions):
            with self.subTest(question=question):
                print(f"\nTest {idx+1}: {question}")

                # Truy vấn với threshold
                results = self.rag_system.query(question, similarity_threshold=TEST_SIMILARITY_THRESHOLD)

                # Lưu kết quả để tạo summary
                result_info = {
                    "query": question,
                    "has_results": len(results) > 0,
                    "result_count": len(results),
                }

                if len(results) > 0:
                    best_match = results[0]
                    result_info["best_match"] = best_match.metadata.get("question", "")
                    result_info["distance"] = best_match.metadata.get("distance", 1.0)

                    print(f"Có {len(results)} kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")
                    print(f"Best match:")
                    print(f"  Question: {best_match.metadata.get('question', '')}")
                    print(f"  Answer:   {best_match.metadata.get('answer', '')}")
                    print(f"  Distance: {best_match.metadata.get('distance', 1.0)}")
                else:
                    print(f"Không có kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")

                summary_results.append(result_info)

        # In summary
        print("\n=== TEST SUMMARY ===")
        print(f"Tổng số câu hỏi kiểm thử: {len(self.test_questions)}")

        successful_queries = sum(1 for r in summary_results if r["has_results"])
        print(f"Số câu hỏi có kết quả: {successful_queries}")
        print(f"Số câu hỏi không có kết quả: {len(self.test_questions) - successful_queries}")

if __name__ == "__main__":
    unittest.main()
