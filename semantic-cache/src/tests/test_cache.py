import unittest
import asyncio
from typing import Any
from pydantic import BaseModel
from semantic_cache_engine.cache import PluggableSemanticCache, QueryAnalysisResult
from semantic_cache_engine.eviction_policies.lru import LRUPolicy
from semantic_cache_engine.eviction_policies.lfu import LFUPolicy

class FunctionalFixedEncoder:
    """Creates mathematically clean, independent unit vectors to simulate distinct keys."""
    def __init__(self):
        self.dimension = 1536

    def embed_query(self, text: str) -> list:
        vector = [0.0] * self.dimension
        # Statically assign indices based on content keywords to guarantee 0.0 similarity
        if "one" in text.lower() or "first" in text.lower():
            vector[0] = 1.0
        elif "two" in text.lower() or "second" in text.lower():
            vector[1] = 1.0
        elif "three" in text.lower() or "third" in text.lower():
            vector[2] = 1.0
        else:
            # Fallback index for generic lookups
            vector[99] = 1.0
        return vector


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(
        self, 
        router_feedback: bool, 
        router_clean_query: str, 
        main_llm_text: str,
        requires_realtime_data: bool = False  # Extended to support dynamic routing tests
    ):
        self.router_feedback = router_feedback
        self.router_clean_query = router_clean_query
        self.main_llm_text = main_llm_text
        self.requires_realtime_data = requires_realtime_data

    def with_structured_output(self, schema: type[BaseModel]):
        return self

    async def ainvoke(self, input_data: Any, **kwargs: Any) -> Any:
        if isinstance(input_data, list) or "system" in str(input_data):
            return QueryAnalysisResult(
                is_negative_feedback=self.router_feedback,
                cleaned_query=self.router_clean_query,
                requires_realtime_data=self.requires_realtime_data  # Mapped cleanly here
            )
        return FakeLLMResponse(content=self.main_llm_text)


# -------------------------------------------------------------------
# UNIT TEST CASES
# -------------------------------------------------------------------
class TestPluggableSemanticCacheUnits(unittest.IsolatedAsyncioTestCase):
    
    async def test_cache_miss_path(self):
        """Verify that a cache miss calls the main LLM and inserts data into the policy."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=False, router_clean_query="query one", main_llm_text="LLM Answer")
        policy = LRUPolicy(max_size=2)
        
        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.85, policy=policy
        )

        response = await cache.process_query("query one")
        self.assertEqual(response, "LLM Answer")
        
        await asyncio.sleep(0.01) # Allow async worker to execute
        self.assertEqual(policy.hit(1)["response"], "LLM Answer")

    async def test_cache_hit_path(self):
        """Verify that a cache hit returns data instantly without calling the main LLM."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=False, router_clean_query="query one", main_llm_text="FAIL")
        policy = LRUPolicy(max_size=2)
        
        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.50, policy=policy
        )

        evicted, assigned_id = policy.put("query one", "Pre-existing Cached Value")
        cache.vector_store.add_vector("query one", assigned_id=assigned_id)

        response = await cache.process_query("query one")
        self.assertEqual(response, "Pre-existing Cached Value")

    async def test_negative_feedback_overwrite_path(self):
        """Verify negative feedback triggers an in-place update when above the threshold."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=True, router_clean_query="query one", main_llm_text="Fresh Corrected Answer")
        policy = LRUPolicy(max_size=2)
        
        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.50, force_update_threshold=0.60, policy=policy
        )

        evicted, assigned_id = policy.put("query one", "Broken Initial Answer")
        cache.vector_store.add_vector("query one", assigned_id=assigned_id)

        response = await cache.process_query("query one")
        self.assertEqual(response, "Fresh Corrected Answer")
        self.assertEqual(policy.hit(1)["response"], "Fresh Corrected Answer")

    async def test_cache_eviction_removes_vector_and_policy(self):
        """Verify that when the LRU cache is full, inserting a new item evicts oldest entries."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=False, router_clean_query="query three", main_llm_text="Value Three")
        policy = LRUPolicy(max_size=2)
        
        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.85, policy=policy
        )

        # Populate capacity strictly (IDs 1 and 2)
        policy.put("query one", "Value One")
        cache.vector_store.add_vector("query one", assigned_id=1)
        policy.put("query two", "Value Two")
        cache.vector_store.add_vector("query two", assigned_id=2)

        # Trigger clean cache miss (Vector index position 2 vs 0 and 1)
        response = await cache.process_query("query three")
        self.assertEqual(response, "Value Three")
        
        await asyncio.sleep(0.01)

        # Confirm ID 1 is purged, and ID 3 successfully populated
        with self.assertRaises(KeyError):
            policy.hit(1)
        
        self.assertEqual(policy.hit(3)["response"], "Value Three")

    async def test_negative_feedback_low_threshold_fallback(self):
        """Verify negative feedback with low similarity inputs inserts a new entry."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=True, router_clean_query="query two", main_llm_text="Distant Correction")
        policy = LRUPolicy(max_size=2)
        
        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.50, force_update_threshold=0.90, policy=policy
        )

        policy.put("query one", "Original Value")
        cache.vector_store.add_vector("query one", assigned_id=1)

        # Query two has 0.0 similarity to query one. Skip overwrite -> insert new vector!
        response = await cache.process_query("query two")
        self.assertEqual(response, "Distant Correction")
        
        await asyncio.sleep(0.01)

        # Confirm both unique records exist simultaneously
        self.assertEqual(policy.hit(1)["response"], "Original Value")
        self.assertEqual(policy.hit(2)["response"], "Distant Correction")

    # -------------------------------------------------------------------
    # NEW EXTENDED DYNAMIC ROUTING TESTS
    # -------------------------------------------------------------------
    async def test_dynamic_routing_enabled_skips_cache(self):
        """Verify that enabled dynamic routing bypasses all cache policy insertions for volatile requests."""
        encoder = FunctionalFixedEncoder()
        # Initialize stub stating that this is a real-time query
        llm = FakeLLM(router_feedback=False, router_clean_query="what time is it", main_llm_text="Live Time: 11:15 PM", requires_realtime_data=True)
        policy = LRUPolicy(max_size=2)

        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.85, policy=policy,
            enable_dynamic_routing=True  # Feature turned ON explicitly
        )

        response = await cache.process_query("What time is it right now?")
        self.assertEqual(response, "Live Time: 11:15 PM")

        await asyncio.sleep(0.01)
        # Verify the invariant: Policy storage remains empty because the cache was completely bypassed
        self.assertEqual(len(policy.get_all_records()), 0)

    async def test_dynamic_routing_disabled_falls_back_to_cache(self):
        """Verify that when dynamic routing is disabled, volatile markers are ignored and cached normally."""
        encoder = FunctionalFixedEncoder()
        llm = FakeLLM(router_feedback=False, router_clean_query="what time is it", main_llm_text="Live Time: 11:15 PM", requires_realtime_data=True)
        policy = LRUPolicy(max_size=2)

        cache = PluggableSemanticCache(
            encoder=encoder, fast_llm=llm, main_llm=llm,
            dimension=1536, hit_threshold=0.85, policy=policy,
            enable_dynamic_routing=False  # Feature turned OFF explicitly
        )

        response = await cache.process_query("What time is it right now?")
        self.assertEqual(response, "Live Time: 11:15 PM")

        await asyncio.sleep(0.01)
        # Verify that it fell back to normal caching: The entry should now exist inside the policy layer
        self.assertEqual(len(policy.get_all_records()), 1)
        self.assertEqual(policy.hit(1)["response"], "Live Time: 11:15 PM")

    def test_invalid_capacity_initialization(self):
        """Verify that specifying a non-positive max_size immediately throws a ValueError."""
        with self.assertRaises(ValueError):
            LFUPolicy(max_size=0)
            
        with self.assertRaises(ValueError):
            LFUPolicy(max_size=-10)

if __name__ == "__main__":
    unittest.main()