import asyncio
from typing import Any
import numpy as np
from loguru import logger

from semantic_cache_engine import PluggableSemanticCache, LRUPolicy
from semantic_cache_engine.interfaces.cache_components import IEncoder, ILargeLanguageModel

# ==========================================
# 1. COMPONENT CONTRACT ALIGNMENT
# ==========================================

class MockEncoder(IEncoder):
    """Generates simple, deterministic mock embeddings for predictable matching."""
    def embed_query(self, text: str) -> np.ndarray:
        # Generate a deterministic vector based on string length rules
        val = (len(text) % 10) / 10.0
        vec = np.array([val, 0.1, 0.5, 0.2], dtype=np.float32)
        return vec / np.linalg.norm(vec)

class MockLLMResponse:
    """Mock container providing a .content string property."""
    def __init__(self, text: str):
        self.content = text

class MockLLM(ILargeLanguageModel):
    """Simulates an LLM implementation supporting structured routing and content blocks."""
    
    async def ainvoke(self, prompt: Any) -> MockLLMResponse:
        # Check if the prompt is an intent routing list or a raw string
        if isinstance(prompt, list):
            logger.info("[LLM Core] Processing structured intent analysis...")
            # Simulate a standard cache route analysis result
            from semantic_cache_engine.cache import QueryAnalysisResult
            return QueryAnalysisResult(
                is_negative_feedback=False, 
                cleaned_query="mock_intent_phrase"
            )
        
        logger.info(f"[LLM Core] Live generating text for: '{prompt}'")
        return MockLLMResponse(f"Generated dynamic response for -> '{prompt}'")

    def with_structured_output(self, schema: Any) -> Any:
        """Returns self to mock out the dynamic intent parsing system."""
        return self

# ==========================================
# 2. RUNTIME SIMULATION SANDBOX
# ==========================================

async def main():
    logger.info("Initializing Sandbox Simulation...")

    encoder = MockEncoder()
    mock_llm = MockLLM()
    
    # 1. Configure policy using your parameter structure ('policy')
    tiny_policy = LRUPolicy(max_size=2)
    
    # 2. Instantiate matching your constructor's exact signatures
    cache = PluggableSemanticCache(
        encoder=encoder,
        fast_llm=mock_llm,
        main_llm=mock_llm,
        dimension=4,                  # Match our mock encoder array dimension
        hit_threshold=0.80,
        policy=tiny_policy,
        enable_dynamic_routing=True
    )

    print("\n--- Step 1: Processing Cold Query (Cache Miss) ---")
    q1 = "Tell me about Transformers in LLM?"
    resp1 = await cache.process_query(q1)
    print(f"Result 1: {resp1}")

    # Yield control to the event loop momentarily to let background task execution finish writes
    await asyncio.sleep(0.1)

    print("\n--- Step 2: Processing Semantic Match Query (Cache Hit) ---")
    # Same vector generation rules apply
    resp2 = await cache.process_query(q1)
    print(f"Result 2: {resp2}")

if __name__ == "__main__":
    asyncio.run(main())