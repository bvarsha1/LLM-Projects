from collections import OrderedDict
from typing import Optional
from threading import Lock
from loguru import logger
from semantic_cache_engine.eviction_policies.base import BaseEvictionPolicy, CachePayload

class LRUPolicy(BaseEvictionPolicy):
    def __init__(self, max_size: int):
        # Prevent structural initialization failure for invalid boundaries
        if max_size <= 0:
            logger.error(f"Initialization Error: Invalid capacity configuration requested: {max_size}")
            raise ValueError("Cache capacity must be a positive integer greater than 0.")
            
        super().__init__(max_size)
        
        # Maps permanent store_key (int) -> TypedDict payload
        self.store: OrderedDict[int, CachePayload] = OrderedDict()
        
        # Guard mapping mutations from parallel thread interleaving
        self._lock: Lock = Lock()

        logger.info(f"LRUPolicy initialized using direct ID-mapping. Capacity: {self.capacity}")

    def is_full(self) -> bool:
        """Helper to determine if the cache data footprint has hit its boundary limits."""
        return len(self.store) >= self.capacity

    def hit(self, store_key: int) -> CachePayload:
        """Retrieves data directly using the permanent vector ID in O(1) time."""
        with self._lock:
            if store_key not in self.store:
                logger.error(f"Routing Error: Vector ID {store_key} does not exist in store.")
                raise KeyError(f"Vector ID {store_key} missing from cache footprint.")

            # Update chronological sequence recency tracking by moving it to the back
            self.store.move_to_end(store_key)
            return self.store[store_key]

    def overwrite(self, store_key: int, new_response: str) -> None:
        """
        Surgically overwrites a payload response in O(1) without changing its ID,
        then updates its chronological sequence position to Most Recently Used (MRU).
        """
        with self._lock:
            if store_key not in self.store:
                logger.error(f"Overwrite Error: Target ID {store_key} does not exist.")
                raise KeyError(f"Cannot overwrite: ID {store_key} missing from memory store.")

            # In-place value substitution
            self.store[store_key]["response"] = new_response
            
            # Shuffle the item to the back of the queue because it was just updated
            self.store.move_to_end(store_key)
            logger.info(f"Policy Overwrite Success: Refreshed text payload for permanent ID {store_key}")

    def put(self, query: str, response: str) -> tuple[Optional[int], int]:
        """
        Inserts new items under capacity boundaries.
        Returns a tuple of (evicted_store_key, assigned_store_key).
        """
        with self._lock:
            evicted_store_key: Optional[int] = None

            # 1. Capacity Audit Layer: Purge the oldest record if limits are breached
            if self.is_full():
                # Extract the oldest key from the very front of the OrderedDict line in O(1)
                evicted_store_key = next(iter(self.store))
                
                # Drop the payload from the dictionary footprint completely
                self.store.pop(evicted_store_key)
                logger.warning(f"Capacity Limit Breached. Policy evicted oldest ID: {evicted_store_key}")

            # 2. Allocation Layer: Secure a permanent identifier anchor
            assigned_store_key = self.next_store_key
            
            # Save record text attributes directly under this identifier
            self.store[assigned_store_key] = {
                "query": query,
                "response": response
            }
            
            # Increment tracking state to secure unique bounds for the next entry
            self.next_store_key += 1
            
            logger.info(f"Policy Put Success: Assigned tracking ID {assigned_store_key}")
            return evicted_store_key, assigned_store_key

    def get_all_records(self) -> dict[int, CachePayload]:
        """Returns the internal raw store mappings for diagnostic or tracking queries."""
        with self._lock:
            return dict(self.store)