import os
import logging
from typing import List, Dict, Tuple, Optional

from ..models.data_models import RAG_DataItem
from .text_adapter import RAG_PlainTextAdapter, RAG_TextItem

class RAG_PDFItem(RAG_TextItem):
    """Cấu trúc dữ liệu dành riêng cho các chunk PDF"""
    page_number: int

    def model_post_init(self, __context):
        """Mở rộng từ TextItem và thêm metadata đặc thù PDF"""
        # Gọi phương thức của lớp cha
        super().model_post_init(__context)
        # Cập nhật metadata với thông tin trang
        self.metadata.update({
            "page_number": self.page_number,
            "name": "pdf_file"
        })

class RAG_PDFAdapter(RAG_PlainTextAdapter):
    """Adapter cho file PDF, kế thừa từ adapter văn bản thuần túy"""

    def __init__(self):
        """Khởi tạo adapter"""
        super().__init__()

    @property
    def name(self) -> str:
        return "pdf_file"

    def load(self,
             file_path: str,
             chunk_size: int = 1000,
             chunk_overlap: int = 200,
             separators: Optional[List[str]] = None) -> List[RAG_DataItem]:
        """
        Load và xử lý file PDF sử dụng PyPDF2

        Args:
            file_path: Đường dẫn đến file PDF
            chunk_size: Kích thước tối đa của mỗi chunk
            chunk_overlap: Độ chồng lấp giữa các chunk
            separators: Danh sách các dấu phân cách (tùy chọn)

        Returns:
            List of RAG_PDFItem objects
        """
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            raise Exception("langchain not installed. Please install it with: pip install langchain")

        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return []

        # Trích xuất văn bản và thông tin trang từ PDF
        extracted_text, page_map = self._extract_pdf_text(file_path)

        if not extracted_text:
            logging.error(f"Failed to extract text from PDF: {file_path}")
            return []

        logging.info(f"Extracted {len(extracted_text)} characters from PDF")

        # --- Phần này tái sử dụng logic từ text_adapter ---

        # Khởi tạo text splitter
        splitter_args = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "length_function": len,
        }

        # Thêm separators nếu có
        if separators:
            splitter_args["separators"] = separators

        text_splitter = RecursiveCharacterTextSplitter(**splitter_args)

        # Phân tách văn bản
        chunks = text_splitter.split_text(extracted_text)
        logging.info(f"Split PDF into {len(chunks)} chunks")

        # --- Phần xử lý đặc thù cho PDF ---

        # Map chunks với thông tin trang
        chunk_page_numbers = self._map_chunks_to_pages(chunks, page_map)

        # Tạo RAG_PDFItem
        documents = []
        for i, chunk in enumerate(chunks):
            file_name = os.path.basename(file_path)
            item_id = f"pdf_{file_name}_{i}"

            # Lấy số trang cho chunk này
            page_number = chunk_page_numbers[i]

            # Tạo RAG_PDFItem
            doc = RAG_PDFItem(
                id=item_id,
                content=chunk,
                chunk_index=i,
                source_file=file_name,
                page_number=page_number,
                metadata={
                    "file_name": file_name,
                    "file_path": file_path,
                }
            )
            documents.append(doc)

        logging.info(f"Created {len(documents)} RAG_PDFItem objects")
        return documents

    def _extract_pdf_text(self, file_path: str) -> Tuple[str, Dict[int, Tuple[int, int]]]:
        """
        Trích xuất văn bản từ file PDF sử dụng PyPDF2

        Returns:
            Tuple[str, Dict[int, Tuple[int, int]]]:
                - Văn bản trích xuất
                - Dictionary map vị trí ký tự với số trang
        """
        try:
            import PyPDF2
        except ImportError:
            raise Exception("PyPDF2 not installed. Please install it with: pip install PyPDF2")

        full_text = ""
        # Map vị trí ký tự với số trang {char_index: (page_number, page_length)}
        page_map = {}

        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                current_position = 0
                for page_num, page in enumerate(reader.pages):
                    # Trích xuất văn bản từ trang
                    page_text = page.extract_text()

                    if page_text:
                        # Thêm phân cách trang
                        if full_text:
                            full_text += "\n\n"
                            current_position += 2

                        # Ghi nhớ vị trí bắt đầu và kết thúc của trang này
                        start_pos = current_position

                        # Cập nhật map vị trí-trang
                        page_map[current_position] = (page_num + 1, len(page_text))

                        # Thêm văn bản vào kết quả
                        full_text += page_text
                        current_position += len(page_text)

            logging.info(f"Extracted text from {len(page_map)} pages")
            return full_text, page_map

        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")
            return "", {}

    def _map_chunks_to_pages(self, chunks: List[str], page_map: Dict[int, Tuple[int, int]]) -> List[int]:
        """
        Map các chunk với số trang tương ứng

        Args:
            chunks: Danh sách các chunk văn bản
            page_map: Dictionary map vị trí ký tự với số trang và độ dài

        Returns:
            List[int]: Danh sách số trang tương ứng với mỗi chunk
        """
        chunk_pages = []

        # Điểm bắt đầu của các trang
        page_starts = sorted(page_map.keys())

        # Vị trí bắt đầu của text
        current_pos = 0

        for chunk in chunks:
            # Tìm trang chứa vị trí bắt đầu của chunk
            page_num = 1  # Mặc định trang đầu tiên

            for start_pos in page_starts:
                if start_pos > current_pos:
                    break
                page_info = page_map[start_pos]
                page_num = page_info[0]  # page_number

                # Kiểm tra xem chunk có nằm trong trang này không
                page_end = start_pos + page_info[1]  # page_length
                if current_pos >= start_pos and current_pos < page_end:
                    break

            chunk_pages.append(page_num)

            # Cập nhật vị trí cho chunk tiếp theo, trừ đi overlap
            # Đây là ước tính thôi, không chính xác 100% do chunking
            current_pos += len(chunk)

        return chunk_pages
