from abc import ABC, abstractmethod
from typing import Optional, TypedDict

class CachePayload(TypedDict):
    query: str
    response: str

class BaseEvictionPolicy(ABC):
    def __init__(self, max_size: int):
        self.capacity: int = max_size
        self.next_store_key: int = 1

    @abstractmethod
    def is_full(self) -> bool:
        """Helper to determine if the cache data footprint has hit its boundary limits."""
        pass

    @abstractmethod
    def hit(self, store_key: int) -> CachePayload:
        """Retrieves data directly using the permanent vector ID in O(1) time."""
        pass

    @abstractmethod
    def overwrite(self, store_key: int, new_response: str) -> None:
        """Surgically overwrites a payload response without changing its ID."""
        pass

    @abstractmethod
    def put(self, query: str, response: str) -> tuple[Optional[int], int]:
        """Inserts new items under capacity boundaries. Returns (evicted_key, assigned_key)."""
        pass

    @abstractmethod
    def get_all_records(self) -> dict[int, CachePayload]:
        """Returns the internal raw store mappings for diagnostic or tracking queries."""
        pass