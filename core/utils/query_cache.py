import os
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple

DEFAULT_CACHE_SIZE = int(os.getenv("QUERY_CACHE_DEFAULT_SIZE") or '1000')
DEFAULT_CACHE_TTL  = int(os.getenv("QUERY_CACHE_DEFAULT_TTL")  or '3600')

class QueryCache:
    """
    Lớp quản lý cache cho các truy vấn RAG.
    Lưu trữ kết quả truy vấn và tái sử dụng khi có truy vấn tương tự.
    """
    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl: int = DEFAULT_CACHE_TTL):
        """
        Khởi tạo bộ nhớ cache.
        
        Args:
            max_size: Kích thước tối đa của cache (số lượng entries)
            ttl: Thời gian sống của cache entries (seconds)
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}  # {key: (value, timestamp)}
        self._max_size = max_size
        self._ttl = ttl
        logging.info(f"Query cache đã được kích hoạt với max_size={max_size}, ttl={ttl}s")

    def _generate_key(self, params: Dict[str, Any]) -> str:
        """
        Tạo khóa cho cache dựa trên các tham số truy vấn.
        
        Args:
            params: Dictionary chứa tất cả tham số, bao gồm query_text
            
        Returns:
            str: Khóa cache dạng hash
        """
        # Kiểm tra query_text là bắt buộc
        if "query_text" not in params:
            raise ValueError("query_text là bắt buộc trong params")
            
        # Tạo bản sao của params để không làm thay đổi params gốc
        cache_params = params.copy()
        
        # Chuẩn hóa query_text
        cache_params["query_text"] = cache_params.get("query_text", "").lower().strip()
        
        # Sắp xếp params để đảm bảo tính nhất quán
        sorted_params = sorted(cache_params.items())
        
        # Tạo chuỗi đại diện cho params
        cache_str = f"{sorted_params}"
        
        # Tạo hash để sử dụng làm key
        return hashlib.sha256(cache_str.encode('utf-8')).hexdigest()

    def get(self, params: Dict[str, Any]) -> Optional[Any]:
        """
        Lấy kết quả từ cache nếu có.
        
        Args:
            params: Dictionary chứa tất cả tham số, bao gồm query_text
            
        Returns:
            Any: Kết quả cache hoặc None nếu không tìm thấy
        """
        key = self._generate_key(params)
        
        # Kiểm tra cache và thời gian sống
        if key in self._cache:
            value, timestamp = self._cache[key]
            current_time = time.time()
            
            # Kiểm tra thời gian sống
            if current_time - timestamp <= self._ttl:
                logging.debug(f"Cache hit for query: {params.get('query_text', '')[:50]}...")
                return value
            else:
                # Xóa entry hết hạn
                logging.debug(f"Cache expired for query: {params.get('query_text', '')[:50]}...")
                del self._cache[key]
        
        logging.debug(f"Cache miss for query: {params.get('query_text', '')[:50]}...")
        return None

    def set(self, params: Dict[str, Any], value: Any) -> None:
        """
        Lưu kết quả vào cache.
        
        Args:
            params: Dictionary chứa tất cả tham số, bao gồm query_text
            value: Kết quả truy vấn cần lưu
        """
        key = self._generate_key(params)
        current_time = time.time()
        
        # Xóa entry cũ nhất nếu đạt max size
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        # Lưu kết quả với timestamp
        self._cache[key] = (value, current_time)
        logging.debug(f"Cached result for query: {params.get('query_text', '')[:50]}...")

    def _evict_oldest(self) -> None:
        """Xóa entry cũ nhất trong cache khi đạt giới hạn kích thước."""
        if not self._cache:
            return
            
        # Tìm key có timestamp cũ nhất
        oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
        del self._cache[oldest_key]
        logging.debug(f"Evicted oldest cache entry with key: {oldest_key}")

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        self._cache.clear()
        logging.info("Cache cleared")

    def statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về cache.
        
        Returns:
            Dict: Thông tin thống kê của cache
        """
        current_time = time.time()
        active_entries = sum(1 for _, timestamp in self._cache.values() 
                            if current_time - timestamp <= self._ttl)
                            
        return {
            "total_entries": len(self._cache),
            "active_entries": active_entries,
            "expired_entries": len(self._cache) - active_entries,
            "max_size": self._max_size,
            "ttl": self._ttl,
            "memory_usage_bytes": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """Ước tính dung lượng bộ nhớ sử dụng bởi cache."""
        # Đây là ước tính đơn giản, không chính xác hoàn toàn
        import sys
        try:
            sample_size = min(10, len(self._cache))
            if sample_size == 0:
                return 0
                
            sample_keys = list(self._cache.keys())[:sample_size]
            avg_key_size = sum(sys.getsizeof(k) for k in sample_keys) / sample_size
            
            sample_values = [self._cache[k][0] for k in sample_keys]
            avg_value_size = sum(sys.getsizeof(v) for v in sample_values) / sample_size
            
            # Cộng thêm kích thước timestamp (float - 8 bytes)
            return int((avg_key_size + avg_value_size + 8) * len(self._cache))
        except:
            return 0