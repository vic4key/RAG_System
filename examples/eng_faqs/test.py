import os
import logging
import unittest

from rag_system import RAG_System
from eng_faqs_adapter import EnglishFAQsAdapter

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging chi tiết hơn
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.7  # Ngưỡng tương đồng cho English FAQs

class TestEnglishFAQsAdapter(unittest.TestCase):
    """Test cases for English FAQs with the RAG System"""

    @classmethod
    def setUpClass(cls):
        """Set up the RAG system once for all test cases"""
        print("\n=== SETTING UP RAG SYSTEM FOR ENGLISH FAQS ===")

        # Set up file path and storage path
        file_path = os.path.join(os.path.dirname(__file__), 'eng_faqs.json')
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        persist_dir = os.path.join(os.getcwd(), f"chroma_db/{file_name_without_ext}")

        # Initialize RAG system
        cls.rag_system = RAG_System(
            endpoint=os.getenv("LLM_EMBEDDING_ENDPOINT"),
            model_name=os.getenv("LLM_EMBEDDING_MODEL_NAME"),
            api_key=os.getenv("LLM_EMBEDDING_API_KEY"),
            api_version=os.getenv("LLM_EMBEDDING_API_VERSION"),
            provider=os.getenv("LLM_EMBEDDING_PROVIDER"),
        )

        # Create adapter
        adapter = EnglishFAQsAdapter()

        # Set up with the adapter using persistent storage
        print(f"Setting up RAG with persistent_directory={persist_dir}, recreate=False")
        cls.rag_system.setup_from_adapter(
            adapter=adapter,
            adapter_params={
                "file_path": file_path,
            },
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,
            recreate=False,
        )

        print(f"Similarity threshold: {TEST_SIMILARITY_THRESHOLD}")
        print("=== SETUP COMPLETE ===\n")

        # Test queries
        cls.test_questions = [
            "How do I create an account?",
            "I forgot my password",
            "How to log out?",
            "Can I use Face ID to log in?",
            "How do I delete my account?",
            "How to change password?",
        ]

    def test_queries(self):
        """Test queries with similarity threshold"""
        
        print("\n=== TESTING QUERIES WITH THRESHOLD ===")
        
        # Store results for summary
        summary_results = []
        
        for idx, question in enumerate(self.test_questions):
            with self.subTest(question=question):
                print(f"\nTest {idx+1}: {question}")
                
                # Query with threshold
                results = self.rag_system.query(question, similarity_threshold=TEST_SIMILARITY_THRESHOLD)
                
                # Store results for summary
                result_info = {
                    "query": question,
                    "has_results": len(results) > 0,
                    "result_count": len(results),
                }
                
                if len(results) > 0:
                    best_match = results[0]
                    result_info["best_match"] = best_match.metadata.get("title", "")
                    result_info["category"] = best_match.metadata.get("category", "")
                    result_info["distance"] = best_match.metadata.get("distance", 1.0)
                    
                    print(f"Found {len(results)} results meeting threshold {TEST_SIMILARITY_THRESHOLD}")
                    print(f"Best match:")
                    print(f"  Category: {best_match.metadata.get('category', '')}")
                    print(f"  Title: {best_match.metadata.get('title', '')}")
                    print(f"  Instruction: {best_match.metadata.get('instruction', '')}")
                    print(f"  Distance: {best_match.metadata.get('distance', 1.0)}")
                else:
                    print(f"No results found meeting threshold {TEST_SIMILARITY_THRESHOLD}")
                
                summary_results.append(result_info)
        
        # Print summary
        print("\n=== TEST SUMMARY ===")
        print(f"Total test questions: {len(self.test_questions)}")
        
        successful_queries = sum(1 for r in summary_results if r["has_results"])
        print(f"Questions with results: {successful_queries}")
        print(f"Questions without results: {len(self.test_questions) - successful_queries}")
        
        if successful_queries > 0:
            # Calculate average distance for questions with results
            avg_distance = sum(r["distance"] for r in summary_results if r["has_results"]) / successful_queries
            print(f"Average distance: {avg_distance:.4f}")
            
            # Question with best result (lowest distance)
            best_result = min((r for r in summary_results if r["has_results"]), key=lambda x: x["distance"])
            print(f"Question with best result: {best_result['query']}")
            print(f"  - Answer: {best_result['best_match']}")
            print(f"  - Category: {best_result['category']}")
            print(f"  - Distance: {best_result['distance']:.4f}")

if __name__ == "__main__":
    unittest.main()
