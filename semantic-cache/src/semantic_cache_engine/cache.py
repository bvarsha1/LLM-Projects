import asyncio
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from loguru import logger

# Fixed Package Imports
from semantic_cache_engine.eviction_policies.base import CachePayload
from semantic_cache_engine.eviction_policies.lru import LRUPolicy
from semantic_cache_engine.eviction_policies.lfu import LFUPolicy
from semantic_cache_engine.vector_store.faiss import FAISSVectorStore
from semantic_cache_engine.interfaces.cache_components import IEncoder, ILargeLanguageModel

class QueryAnalysisResult(BaseModel):
    is_negative_feedback: bool = Field(
        description="True if user flags frustration or demands response changes."
    )
    cleaned_query: str = Field(
        description="The underlying dense keyword-only search intent."
    )
    requires_realtime_data: bool = Field(
        default=False,
        description="True if the query depends on real-time, volatile data (e.g., time, weather, stocks)."
    )

# Detailed systemic prompt to eliminate model non-determinism
SYSTEM_ROUTER_PROMPT = (
    "You are a strict cache routing agent. Analyze the user prompt to extract:\n"
    "1. is_negative_feedback (boolean): True if the user complains, states an answer was wrong, or wants a retry.\n"
    "2. cleaned_query (string): Core intent keyword phrase, stripped of conversation filler or greetings."
    "3. requires_realtime_data (boolean): True ONLY if the query depends on highly volatile, real-time facts "
    "such as the current time, today's date, live weather conditions, or real-time stock prices."
)

class PluggableSemanticCache:
    def __init__(
        self, 
        encoder: IEncoder,
        fast_llm: ILargeLanguageModel,
        main_llm: ILargeLanguageModel,
        dimension: int = 1536, 
        hit_threshold: float = 0.88,
        force_update_threshold: float = 0.95,
        policy: Optional[LRUPolicy] = None,
        index_type: Literal["flat", "ivf", "hnsw"] = "flat",
        training_data: Any = None,
        enable_dynamic_routing: bool = False,
    ):
        self.main_llm = main_llm  
        self.hit_threshold = hit_threshold
        self.force_update_threshold = force_update_threshold
        
        # The single synchronization lock that protects all background mutations
        self._async_lock = asyncio.Lock()
        self.enable_dynamic_routing = enable_dynamic_routing
        
        self.policy = policy if policy is not None else LRUPolicy(max_size=100)
        self.vector_store = FAISSVectorStore(encoder, dimension, index_type, training_data)
        self.analyzer_llm = fast_llm.with_structured_output(QueryAnalysisResult)

    async def _analyze_intent_and_clean(self, raw_text: str) -> QueryAnalysisResult:
        try:
            return await self.analyzer_llm.ainvoke([
                {"role": "system", "content": SYSTEM_ROUTER_PROMPT},
                {"role": "user", "content": raw_text}
            ])
        except Exception:
            return QueryAnalysisResult(is_negative_feedback=False, cleaned_query=raw_text)

    def _execute_synchronized_write(self, query_str: str, response: str) -> None:
        """Linear mutation execution path. Only called inside the lock worker."""
        evicted_id, assigned_id = self.policy.put(query_str, response)

        if evicted_id is not None:
            self.vector_store.delete_vector(target_store_key=evicted_id)

        self.vector_store.add_vector(query_str, assigned_id)

    async def _async_background_write_worker(self, query_str: str, response: str) -> None:
        """Unified background worker handling serialization safety for all misses/retries."""
        async with self._async_lock:
            self._execute_synchronized_write(query_str, response)

    async def process_query(self, raw_user_query: str) -> str:
        analysis = await self._analyze_intent_and_clean(raw_user_query)

        if self.enable_dynamic_routing and getattr(analysis, "requires_realtime_data", False):
            logger.warning(f"Volatile data intent detected for '{raw_user_query}'. Bypassing semantic cache.")
            # Call your main LLM directly using the correct argument and extract .content
            llm_call = await self.main_llm.ainvoke(raw_user_query)
            return llm_call.content

        best_score, best_id = self.vector_store.search_closest(analysis.cleaned_query)

        # Branch A: Negative Feedback Loop Identified
        if analysis.is_negative_feedback:
            if best_id != -1 and best_score >= self.force_update_threshold:
                llm_call = await self.main_llm.ainvoke(raw_user_query)
                fresh_response = llm_call.content
                
                # In-place overwrites are O(1) dictionary swaps. No lock required.
                self.policy.overwrite(store_key=best_id, new_response=fresh_response)
                return fresh_response
            else:
                # Proximity is too low. Treat it as a unique entry.
                llm_call = await self.main_llm.ainvoke(raw_user_query)
                fresh_response = llm_call.content
                
                # Offload to background task
                asyncio.create_task(self._async_background_write_worker(analysis.cleaned_query, fresh_response))
                return fresh_response

        # Branch B: Standard Cache Hit
        if best_id != -1 and best_score >= self.hit_threshold:
            record: CachePayload = self.policy.hit(best_id)
            return record["response"]

        # Branch C: Standard Cache Miss
        llm_call = await self.main_llm.ainvoke(raw_user_query)
        fresh_response = llm_call.content
        
        # Offload to background task
        asyncio.create_task(self._async_background_write_worker(analysis.cleaned_query, fresh_response))
        return fresh_response