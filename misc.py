import logging
from typing import Dict, Any
from openai import AzureOpenAI

def llm_generate_response(client: AzureOpenAI, model_name: str, query: str, best_match: Dict[str, Any]) -> str:
    """Tạo phản hồi dựa trên kết quả truy vấn.

    Args:
        client: Client AzureOpenAI để gọi LLM
        query: Câu hỏi của người dùng
        best_match: Kết quả truy vấn tốt nhất từ vector DB
        model_name: Tên của mô hình LLM được sử dụng

    Returns:
        str: Câu trả lời được tạo bởi LLM
    """
    try:
        if not best_match:
            return "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn. Vui lòng thử lại với câu hỏi khác."

        # Tạo prompt cho LLM để tạo phản hồi tự nhiên hơn
        prompt = f"""Dựa trên câu hỏi của người dùng: "{query}"

Thông tin từ cơ sở dữ liệu:
Câu hỏi tương tự: {best_match['question']}
Câu trả lời: {best_match['answer']}

Hãy trả lời câu hỏi của người dùng một cách tự nhiên, sử dụng thông tin từ cơ sở dữ liệu. Nếu câu trả lời có vẻ không phù hợp, hãy nói rằng bạn không chắc chắn."""

        # Gọi LLM để tạo phản hồi
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI của ứng dụng PayTrackAI, một ứng dụng theo dõi chi tiêu. Nhiệm vụ của bạn là trả lời các câu hỏi về cách sử dụng ứng dụng."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Lỗi khi tạo phản hồi: {e}")
        return "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."
    except Exception as e:
        logging.error(f"Lỗi khi tạo phản hồi: {e}")
        return "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."