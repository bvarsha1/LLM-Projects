from abc import ABC, abstractmethod
from typing import List, Any, Dict
from pydantic import BaseModel

class AIMessageChunk(BaseModel):
    """Normalized minimum payload structure representing LLM execution returns."""
    content: str
    additional_kwargs: Dict[str, Any] = {}


class IEncoder(ABC):
    """
    Contract for embedding models. Any vector calculation middleware 
    must inherit from this layout.
    """
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Converts a textual string into a one-dimensional array of 
        high-dimensional floating-point vector coordinates.
        """
        pass


class ILargeLanguageModel(ABC):
    """
    Contract for Language Models (both fast routers and main execution engines).
    """
    @abstractmethod
    async def ainvoke(self, input_data: Any, **kwargs: Any) -> AIMessageChunk:
        """
        Asynchronously handles execution chains, processing user prompts or message list 
        payload blocks and returning standardized text containers.
        """
        pass

    @abstractmethod
    def with_structured_output(self, schema: type[BaseModel]) -> "ILargeLanguageModel":
        """
        Binds a dynamic target Pydantic evaluation blueprint model to the active LLM context,
        guaranteeing JSON response conformity on subsequent execution pathways.
        """
        pass