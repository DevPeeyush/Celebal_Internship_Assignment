import time
from functools import wraps
from utils.logger import get_logger

logger = get_logger(__name__)

class Metrics:
    def __init__(self):
        self.doc_count = 0
        self.chunk_count = 0
        self.vector_db_size = 0
        self.last_retrieval_latency = 0.0
        self.last_generation_latency = 0.0

metrics_instance = Metrics()

def track_latency(metric_name):
    """Decorator to track execution latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            latency = end_time - start_time
            
            logger.info(f"{func.__name__} completed in {latency:.4f} seconds.")
            
            if metric_name == "retrieval":
                metrics_instance.last_retrieval_latency = latency
            elif metric_name == "generation":
                metrics_instance.last_generation_latency = latency
                
            return result
        return wrapper
    return decorator
