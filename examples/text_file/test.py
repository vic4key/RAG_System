import os
import logging
import unittest

from RAG_System import RAG_System
from RAG_System.core.config import EmbConfig
from RAG_System.core.adapters.text_adapter import RAG_PlainTextAdapter

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TEST_SIMILARITY_THRESHOLD = 0.7  # Similarity threshold for retrieval
CHUNK_SIZE = 150  # Size of text chunks - reduced to fit small text file
CHUNK_OVERLAP = 30   # Overlap between chunks - reduced to be proportional to chunk size

class TestPlainTextAdapter(unittest.TestCase):
    """Test cases for Plain Text Adapter with the RAG System"""

    @classmethod
    def setUpClass(cls):
        """Set up the RAG system once for all test cases"""
        print("\n=== SETTING UP RAG SYSTEM FOR PLAIN TEXT FILE ===")

        # Set up file path and storage path
        file_path = os.path.join(os.path.dirname(__file__), 'my_text_file.txt')
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

        # Initialize RAG system
        cls.rag_system = RAG_System(emb_config=emb_config)

        # Create adapter
        adapter = RAG_PlainTextAdapter()

        # Set up with the provider using persistent storage
        print(f"Setting up RAG with persistent_directory={persist_dir}, recreate=False")

        # Set up the RAG system
        cls.rag_system.setup_from_adapter(
            adapter=adapter,
            adapter_params={
                "file_path": file_path,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,
            recreate=False,
        )

        print(f"Similarity threshold: {TEST_SIMILARITY_THRESHOLD}")
        print("=== SETUP COMPLETE ===\n")

        # Test queries
        cls.test_questions = [
            "What is FQMS?",
            "What is the Quality Policy according to VNOS?",
            "Where can I find quality documents?",
            "Who is responsible for QDS?",
            "What is CMMi?",
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
                    chunk_text = best_match.content
                    result_info["chunk_text"] = chunk_text
                    result_info["distance"] = best_match.metadata.get("distance", 1.0)

                    print(f"Found {len(results)} results meeting threshold {TEST_SIMILARITY_THRESHOLD}")
                    print(f"Best match (chunk {best_match.metadata.get('chunk_index', 'unknown')}):")
                    print(f"  Text:\n```\n{chunk_text.strip()}\n```")
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
            print(f"  - Text: {best_result['chunk_text'][:100]}...")
            print(f"  - Distance: {best_result['distance']:.4f}")

if __name__ == "__main__":
    unittest.main()