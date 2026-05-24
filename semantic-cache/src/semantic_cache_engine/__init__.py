import sys
from loguru import logger

from semantic_cache_engine.cache import PluggableSemanticCache, QueryAnalysisResult
from semantic_cache_engine.eviction_policies.lru import LRUPolicy
from semantic_cache_engine.eviction_policies.lfu import LFUPolicy

__all__ = [
    "PluggableSemanticCache",
    "QueryAnalysisResult",
    "LRUPolicy",
    "LFUPolicy"
]

# 1. Clear default handlers to prevent duplicate console printing
logger.remove()

# 2. Add standard console (terminal) printing with nice colors
logger.add(sys.stderr, level="INFO")

# 3. Add your explicit permanent file logger
logger.add(
    "logs/semantic_cache_engine.log", 
    rotation="10 MB",     # Automatically splits into a new file when it hits 10MB
    retention="10 days",  # Deletes files older than 10 days to save disk space
    compression="zip",    # Compresses old logs to save space
    level="INFO",         # Only capture INFO, WARNING, and ERROR logs
    enqueue=True          # Makes logging fully asynchronous and thread-safe!
)