import os
import logging
import unittest

from rag_system import RAG_System
from core.adapters.pdf_adapter import RAG_PDFAdapter

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.70  # Ngưỡng tương đồng cho PDF

class TestPDFAdapter(unittest.TestCase):
    """Test cases cho PDF Adapter với RAG System"""

    @classmethod
    def setUpClass(cls):
        """Thiết lập hệ thống RAG một lần cho tất cả các test case"""
        print("\n=== THIẾT LẬP HỆ THỐNG RAG CHO PDF TEST ===")

        # Thiết lập tên file và đường dẫn lưu trữ
        file_path = os.path.join(os.path.dirname(__file__), 'my_pdf_file.pdf')
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = os.path.join(os.getcwd(), f"chroma_db/{file_name_without_ext}")

        # Khởi tạo RAG system
        cls.rag_system = RAG_System(
            endpoint=os.getenv("LLM_ENDPOINT"),
            model_name=os.getenv("LLM_EMBEDDING_MODEL_NAME"),
            api_key=os.getenv("LLM_EMBEDDING_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            provider=os.getenv("LLM_PROVIDER"),
        )

        # Tạo PDF adapter
        adapter = RAG_PDFAdapter()

        # Sử dụng persistent database để lưu trữ trên đĩa
        print(f"Thiết lập RAG với persistent_directory={persist_dir}, recreate=False")
        cls.rag_system.setup_from_adapter(
            adapter=adapter,
            adapter_params={
                "file_path": file_path,
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,
            recreate=False,
        )

        print(f"Ngưỡng tương đồng (SIMILARITY_THRESHOLD): {TEST_SIMILARITY_THRESHOLD}")
        print("=== THIẾT LẬP HOÀN TẤT ===\n")

        # Các câu hỏi thử nghiệm (thay đổi phù hợp với nội dung của PDF)
        cls.test_questions = [
            "ALFRED THE GREAT, King of England (849-901)",
            "ETHELRED II, King of England (c. 968-1016)",
            "WILLIAM THE CONQUEROR, King of England (1027-1087)",
            "HENRY I, King of England (1068-1135)",
            "RICHARD I, King of England (1157-99)",
        ]

    def test_pdf_queries(self):
        """Test truy vấn PDF với ngưỡng tương đồng"""

        print("\n=== KIỂM TRA TRUY VẤN PDF ===")

        # Lưu kết quả để tạo summary
        summary_results = []

        for idx, question in enumerate(self.test_questions):
            with self.subTest(question=question):
                print(f"\nTest {idx+1}: {question}")

                # Truy vấn với threshold
                results = self.rag_system.query(question, similarity_threshold=TEST_SIMILARITY_THRESHOLD)

                # Lưu kết quả cho summary
                result_info = {
                    "query": question,
                    "has_results": len(results) > 0,
                    "result_count": len(results),
                }

                if len(results) > 0:
                    best_match = results[0]
                    result_info["content"] = best_match.content[:100] + "..." if len(best_match.content) > 100 else best_match.content
                    result_info["page"] = best_match.metadata.get("page_number", "N/A")
                    result_info["distance"] = best_match.metadata.get("distance", 1.0)

                    print(f"Có {len(results)} kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")
                    print(f"Best match:")
                    print(f"  Page: {best_match.metadata.get('page_number', 'N/A')}")
                    print(f"  Content (first 150 chars):\n```\n{best_match.content[:150].strip()}\n```")
                    print(f"  Distance: {best_match.metadata.get('distance', 1.0)}")
                else:
                    print(f"Không có kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")

                summary_results.append(result_info)

        # In summary
        print("\n=== TÓM TẮT KẾT QUẢ ===")
        print(f"Tổng số câu hỏi kiểm thử: {len(self.test_questions)}")

        successful_queries = sum(1 for r in summary_results if r["has_results"])
        print(f"Số câu hỏi có kết quả: {successful_queries}")
        print(f"Số câu hỏi không có kết quả: {len(self.test_questions) - successful_queries}")

if __name__ == "__main__":
    unittest.main()