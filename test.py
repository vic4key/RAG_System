import unittest
import os
import json
import logging
from rag_system import RAG_System

TEST_SIMILARITY_THRESHOLD = 0.70

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestRAGSystem(unittest.TestCase):
    """Test cases cho RAG System, bao gồm cả tiếng Anh và tiếng Việt, chỉ test method query"""
    
    @classmethod
    def setUpClass(cls):
        """Thiết lập hệ thống RAG một lần cho tất cả các test case"""
        print("\n=== THIẾT LẬP HỆ THỐNG RAG CHO TEST ===")
        
        # Thiết lập tên file và đường dẫn lưu trữ
        file_path = 'sample_faqs.json'
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = f"./chroma_db_test/{file_name_without_ext}"
        
        cls.rag_system = RAG_System()
        
        # Sử dụng persistent database để lưu trữ trên đĩa
        print(f"Thiết lập RAG với persistent_directory={persist_dir}, recreate=False")
        cls.rag_system.setup_from_file(
            file_path=file_path,
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,  # Sử dụng persistent database cho test
            recreate=False  # Sử dụng database hiện có nếu đã tồn tại
        )
        
        # Lưu các câu hỏi gốc để so sánh trong quá trình test
        with open(file_path, 'r', encoding='utf-8') as f:
            cls.original_data = json.load(f)
            cls.faqs = cls.original_data.get('faqs', [])
            
        print(f"Ngưỡng tương đồng (SIMILARITY_THRESHOLD): {TEST_SIMILARITY_THRESHOLD}")
        print("=== THIẾT LẬP HOÀN TẤT ===\n")
        
        # Kiểm thử với một vài câu hỏi
        cls.test_questions = [
            "Làm sao để chỉnh sửa hoặc xóa chi tiêu?",
            "Xóa chi tiêu?",
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
                    result_info["best_match"] = best_match["question"]
                    result_info["distance"] = best_match["distance"]
                    
                    print(f"Có {len(results)} kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")
                    print(f"Best match:")
                    print(f"  Question: {best_match['question']}")
                    print(f"  Answer:   {best_match['answer']}")
                    print(f"  Distance: {best_match['distance']}")
                else:
                    print(f"Không có kết quả đạt ngưỡng tương đồng {TEST_SIMILARITY_THRESHOLD}")
                
                summary_results.append(result_info)
        
        # In summary
        print("\n=== TEST SUMMARY ===")
        print(f"Tổng số câu hỏi kiểm thử: {len(self.test_questions)}")
        
        successful_queries = sum(1 for r in summary_results if r["has_results"])
        print(f"Số câu hỏi có kết quả: {successful_queries}")
        print(f"Số câu hỏi không có kết quả: {len(self.test_questions) - successful_queries}")
        
        if successful_queries > 0:
            # Tính trung bình khoảng cách cho các câu hỏi có kết quả
            avg_distance = sum(r["distance"] for r in summary_results if r["has_results"]) / successful_queries
            print(f"Khoảng cách trung bình: {avg_distance:.4f}")
            
            # Câu hỏi có kết quả tốt nhất (khoảng cách thấp nhất)
            best_result = min((r for r in summary_results if r["has_results"]), key=lambda x: x["distance"])
            print(f"Câu hỏi có kết quả tốt nhất: {best_result['query']}")
            print(f"  - Câu trả lời: {best_result['best_match']}")
            print(f"  - Khoảng cách: {best_result['distance']:.4f}")


if __name__ == "__main__":
    unittest.main()
