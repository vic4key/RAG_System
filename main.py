import os
import logging
from rag_system import RAG_System
from openai import AzureOpenAI

from misc import llm_generate_response

# Tải biến môi trường
from dotenv import load_dotenv
load_dotenv()

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
AZURE_API_VERSION = "2024-07-01-preview"

def run_interactive_mode(rag_system: RAG_System, client: AzureOpenAI, model_name: str):
    """Chạy chế độ tương tác với người dùng."""
    print("\n=== PayTrackAI FAQ Chatbot ===")
    print("Nhập 'quit', 'exit', hoặc 'q' để thoát.\n")
    
    while True:
        query = input("\nNhập câu hỏi của bạn: ")
        query = query.strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Cảm ơn bạn đã sử dụng PayTrackAI FAQ Chatbot!")
            break

        if not query:
            continue

        print("\nĐang tìm kiếm câu trả lời...")

        # Truy vấn dữ liệu
        retrieved_items = rag_system.query(query)

        # In kết quả chi tiết (để debug, có thể comment lại trong sản phẩm thực tế)
        if retrieved_items:
            print("\nKết quả tìm kiếm:")
            for i, item in enumerate(retrieved_items):
                print(f"{i+1}. Câu hỏi: {item['question']}")
                print(f"   Điểm tương đồng: {1 - item['distance']:.4f}")
                print(f"   Câu trả lời: {item['answer']}\n")

        # Tạo phản hồi với client được truyền vào
        response = llm_generate_response(
            client,
            model_name,
            query, retrieved_items[0] if retrieved_items else None,
        )

        print("\n=== Câu trả lời ===")
        print(response)
        print("==================\n")


def main():
    """Hàm chính để chạy ứng dụng."""
    try:
        print("Đang khởi tạo PayTrackAI FAQ Chatbot...")
        
        # Lấy tên model từ biến môi trường
        model_name = os.getenv("LLM_MODEL_NAME")
        if not model_name:
            logging.error("Thiếu biến môi trường LLM_MODEL_NAME")
            raise ValueError("Thiếu biến môi trường LLM_MODEL_NAME")

        # Khởi tạo hệ thống RAG
        rag_system = RAG_System()
        
        # Khởi tạo client LLM riêng biệt
        client = AzureOpenAI(
            azure_endpoint=os.getenv("LLM_ENDPOINT"),
            api_key=os.getenv("LLM_API_KEY"),
            api_version=AZURE_API_VERSION,
        )
        
        # Thiết lập tên collection dựa trên tên file
        file_path = 'sample_faqs.json'
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        # Thiết lập đường dẫn lưu trữ dựa trên tên file
        persist_dir = f"./chroma_db/{file_name_without_ext}"
        
        # Thiết lập hệ thống từ file JSON
        # Lưu ý: Đường dẫn lưu trữ được tạo theo tên file
        # Hoặc persist_directory=None để lưu trong RAM
        rag_system.setup_from_file(
            file_path=file_path, 
            collection_name=file_name_without_ext,
            persist_directory=persist_dir,  # Thay đổi thành None để lưu trong RAM
        )
        
        # Chạy chế độ tương tác với client LLM và model_name được truyền vào
        run_interactive_mode(rag_system, client, model_name)

    except Exception as e:
        logging.error(f"Lỗi khi chạy ứng dụng: {e}")
        print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    main()
