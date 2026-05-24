from collections import OrderedDict
from typing import Optional
from threading import RLock
from loguru import logger
from semantic_cache_engine.eviction_policies.base import BaseEvictionPolicy, CachePayload

class LFUPolicy(BaseEvictionPolicy):
    def __init__(self, max_size: int):
        # Prevent structural initialization failure for invalid boundaries
        if max_size <= 0:
            logger.error(f"Initialization Error: Invalid capacity configuration requested: {max_size}")
            raise ValueError("Cache capacity must be a positive integer greater than 0.")
            
        super().__init__(max_size)
        
        # Guard against re-entrancy deadlocks across nested helper paths
        self._lock: RLock = RLock()
        
        # Core identification mapping
        self.store: dict[int, CachePayload] = {}
        
        # Track frequency counters: store_key -> hit_count
        self.frequencies: dict[int, int] = {}
        
        # Track frequency groupings for O(1) eviction tie-breaking: frequency -> OrderedDict() of keys
        self.freq_groups: dict[int, OrderedDict[int, bool]] = {}
        
        # Track the absolute minimum frequency across the whole cache framework
        self.min_frequency: int = 0
        
        logger.info(f"LFUPolicy initialized with frequency grouping structures. Capacity: {self.capacity}")

    def is_full(self) -> bool:
        return len(self.store) >= self.capacity

    def _increment_frequency(self, store_key: int) -> None:
        """Internal helper to safely elevate frequency states for hit tracking in O(1)."""
        with self._lock:  # Securely re-entrant under the current thread execution context
            current_freq = self.frequencies[store_key]
            next_freq = current_freq + 1
            
            # Update lookup metrics
            self.frequencies[store_key] = next_freq
            
            # Pop from current frequency tracking queue
            self.freq_groups[current_freq].pop(store_key)
            
            # Clean up empty frequency tiers to prevent memory creep
            if not self.freq_groups[current_freq]:
                self.freq_groups.pop(current_freq)
                # Adjust global min_frequency pointer if it was affected
                if self.min_frequency == current_freq:
                    self.min_frequency = next_freq

            # Push to next frequency tier
            if next_freq not in self.freq_groups:
                self.freq_groups[next_freq] = OrderedDict()
            self.freq_groups[next_freq][store_key] = True

    def hit(self, store_key: int) -> CachePayload:
        with self._lock:
            if store_key not in self.store:
                logger.error(f"Routing Error: Vector ID {store_key} does not exist in LFU store.")
                raise KeyError(f"Vector ID {store_key} missing from LFU cache footprint.")
            
            self._increment_frequency(store_key)
            return self.store[store_key]

    def overwrite(self, store_key: int, new_response: str) -> None:
        with self._lock:
            if store_key not in self.store:
                logger.error(f"Overwrite Error: Target ID {store_key} does not exist in LFU store.")
                raise KeyError(f"Cannot overwrite: ID {store_key} missing from LFU memory store.")
            
            self.store[store_key]["response"] = new_response
            # Overwriting counts as an interaction, which increases its access frequency
            self._increment_frequency(store_key)
            logger.info(f"Policy Overwrite Success: Refreshed text payload for permanent ID {store_key}")

    def put(self, query: str, response: str) -> tuple[Optional[int], int]:
        with self._lock:
            evicted_store_key: Optional[int] = None

            if self.is_full():
                # 1. Select the lowest frequency group line
                target_tier = self.freq_groups[self.min_frequency]
                
                # 2. Extract the frontmost item (the oldest item in the tie group break)
                evicted_store_key = next(iter(target_tier))
                
                # 3. Clean up all mapping traces across metadata layers
                target_tier.pop(evicted_store_key)
                if not target_tier:
                    self.freq_groups.pop(self.min_frequency)
                
                self.store.pop(evicted_store_key)
                self.frequencies.pop(evicted_store_key)
                logger.warning(f"LFU Capacity Limit Breached. Evicted ID: {evicted_store_key} (Freq: {self.min_frequency})")

            # Reset minimum tracked baseline entry pointer back to 1 for the incoming item
            self.min_frequency = 1
            assigned_store_key = self.next_store_key
            
            # Persist payload allocations
            self.store[assigned_store_key] = {"query": query, "response": response}
            self.frequencies[assigned_store_key] = 1
            
            if 1 not in self.freq_groups:
                self.freq_groups[1] = OrderedDict()
            self.freq_groups[1][assigned_store_key] = True
            
            self.next_store_key += 1
            logger.info(f"Policy Put Success: Assigned tracking ID {assigned_store_key} in LFU footprint.")
            return evicted_store_key, assigned_store_key

    def get_all_records(self) -> dict[int, CachePayload]:
        with self._lock:
            return dict(self.store)