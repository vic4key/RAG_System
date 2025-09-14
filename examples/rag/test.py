import os
import logging
import unittest

from RAG_System import RAG_System
from RAG_System.examples.sample_faqs.sample_faqs_adapter import SampleFAQsAdapter

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.70

class TestRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # Thiết lập tên file và đường dẫn lưu trữ
        file_path = os.path.join(os.path.dirname(__file__), 'sample_faqs.json')
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = os.path.join(os.getcwd(), f"chroma_db/{file_name_without_ext}")

        # Set up with the provider using persistent storage
        print(f"Setting up RAG with persistent_directory={persist_dir}, recreate=False")
        cls.rag_system = RAG_System()

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
            "Xóa chi tiêu?",
        ]

    def test_llm_generation(self):
        print("\n=== LLM GENERATION WITHOUT STREAMING ===")
        for question in self.test_questions:
            result_items = self.rag_system.query(question)
            print(f"\Retrieval:\n```\n{result_items}\n```")
            answer = self.rag_system.generate(question, result_items, stream=False)
            print(f"\nGeneration:\n```\n{answer}\n```")
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 0)

    def test_llm_generation_streaming(self):
        print("\n=== LLM GENERATION WITH STREAMING ===")
        for question in self.test_questions:
            result_items = self.rag_system.query(question)
            print(f"\Retrieval:\n```\n{result_items}\n```")
            print(f"\nGeneration:")
            for chunk in self.rag_system.generate(question, result_items, stream=True):
                print(chunk, end="", flush=True)
            print()

if __name__ == "__main__":
    unittest.main()
