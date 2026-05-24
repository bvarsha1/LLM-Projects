import numpy as np
import faiss
from typing import Any, Optional, Tuple, Literal
from loguru import logger

class FAISSVectorStore:
    def __init__(
        self, 
        encoder: Any, 
        dimension: int, 
        index_type: Literal["flat", "ivf", "hnsw"], 
        training_data: Optional[np.ndarray] = None
    ):
        self.encoder = encoder
        self.dimension = dimension
        
        # Call the standalone builder function and assign the result
        self._index = self._build_faiss_index(index_type, training_data)
        logger.info(f"VectorStore: Fully initialized with underlying storage engine.")

    def _build_faiss_index(self, index_type: str, training_data: Optional[np.ndarray]) -> faiss.IndexIDMap2:
        """
        Internal builder function handling the direct execution layer.
        Isolates the raw FAISS configuration logic from the constructor.
        """
        strategy = index_type.lower()
        
        if strategy == "flat":
            logger.info("VectorStore: Assembling clean Flat IP index.")
            base_index = faiss.IndexFlatIP(self.dimension)
            
        elif strategy == "hnsw":
            logger.info("VectorStore: Assembling multi-layer HNSW graph index.")
            hnsw_index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            hnsw_index.hnsw.efSearch = 64
            hnsw_index.hnsw.efConstruction = 64
            base_index = hnsw_index
            
        elif strategy == "ivf":
            logger.info("VectorStore: Assembling and training IVF clustered index.")
            quantizer = faiss.IndexFlatIP(self.dimension)
            ivf_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT)
            ivf_index.nprobe = 10
            
            if training_data is None:
                logger.warning("VectorStore: No training data provided for IVF. Bootstrapping mock distribution.")
                training_data = np.random.randn(1000, self.dimension).astype('float32')
                faiss.normalize_L2(training_data)
                
            ivf_index.train(training_data)
            base_index = ivf_index
            
        else:
            raise ValueError(f"VectorStore Error: Index strategy '{index_type}' is not recognized.")
        
        # Wrap with the ID Mapping layer to allow dynamic custom ID assignments
        return faiss.IndexIDMap2(base_index)

    def _prepare_vector(self, text: str) -> np.ndarray:
        """FAISS WORKFLOW: Encapsulates embedding, type casting, and L2 normalization."""
        raw_embedding = self.encoder.embed_query(text)
        vector = np.array(raw_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    def search_closest(self, query_text: str) -> Tuple[float, int]:
        """Accepts a clean string, returns pure Python primitives (score, id)."""
        if self._index.ntotal == 0:
            return -1.0, -1
            
        vector = self._prepare_vector(query_text)
        scores, indices = self._index.search(vector, k=1)
        return float(scores[0][0]), int(indices[0][0])

    def add_vector(self, query_text: str, assigned_id: int) -> None:
        """Saves a query vector tagged with our permanent tracking ID."""
        vector = self._prepare_vector(query_text)
        id_array = np.array([assigned_id], dtype=np.int64)
        self._index.add_with_ids(vector, id_array)

    def delete_vector(self, target_store_key: int) -> None:
        """Surgically purges a vector by its ID in O(1) without rebuilding the matrix."""
        id_selector = faiss.IDSelectorArray(np.array([target_store_key], dtype=np.int64))
        self._index.remove_ids(id_selector)