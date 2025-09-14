import os
import time
import logging
import unittest

from rag_system import RAG_System
from examples.sample_faqs.sample_faqs_adapter import SampleFAQsAdapter

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.70  # Ngưỡng tương đồng cho FAQs

class TestCaching(unittest.TestCase):
    """Test cases cho tính năng caching trong RAG System với Sample FAQs"""

    @classmethod
    def setUpClass(cls):
        """Thiết lập hệ thống RAG một lần cho tất cả các test case"""
        file_path = os.path.join(os.path.dirname(__file__), 'sample_faqs.json')
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = os.path.join(os.getcwd(), f"chroma_db/{file_name_without_ext}")

        cls.rag_system = RAG_System(
            endpoint=os.getenv("LLM_ENDPOINT"),
            model_name=os.getenv("LLM_EMBEDDING_MODEL_NAME"),
            api_key=os.getenv("LLM_EMBEDDING_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            provider=os.getenv("LLM_PROVIDER")
        )

        adapter = SampleFAQsAdapter()

        cls.rag_system.setup_from_adapter(
            adapter=adapter,
            adapter_params={"file_path": file_path},
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,
            recreate=False,
        )

        cls.test_questions = [
            "Làm sao để chỉnh sửa hoặc xóa chi tiêu?",
            "xóa chi tiêu?",
            "How do I connect my bank account?",
            "Can I scan my receipts?",
            "How to scan a receipt?",
        ]

    def test_cache_hit(self):
        """Test cache hit: truy vấn 2 lần liên tiếp, lần 2 phải nhanh hơn và kết quả giống nhau"""
        print("\n=== TEST QUERY CACHE HIT ===")
        query = "Làm sao để chỉnh sửa hoặc xóa chi tiêu?"
        t1 = time.time()
        res1 = self.rag_system.query(query, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
        t2 = time.time()
        res2 = self.rag_system.query(query, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
        t3 = time.time()
        self.assertEqual(len(res1), len(res2))
        if res1 and res2:
            self.assertEqual(res1[0].id, res2[0].id)
        time1 = t2-t1
        time2 = t3-t2
        print(f"Cache miss: {time1:.4f}s, cache hit: {time2:.4f}s")
        self.assertLess(time2, time1)

    def test_cache_params(self):
        """Test cache key phân biệt theo tham số truy vấn"""
        print("\n=== TEST QUERY CACHE WITH DIFF PARAMS ===")
        query = "How do I connect my bank account?"
        res1 = self.rag_system.query(query, top_k=2, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
        res2 = self.rag_system.query(query, top_k=3, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
        self.assertNotEqual(res1, res2)

    def test_multiple_queries(self):
        """Test cache hoạt động với nhiều truy vấn khác nhau"""
        print("\n=== TEST QUERY CACHE WITH MULTIPLE QUERIES ===")
        for query in self.test_questions:
            res1 = self.rag_system.query(query, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
            res2 = self.rag_system.query(query, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
            self.assertEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()
