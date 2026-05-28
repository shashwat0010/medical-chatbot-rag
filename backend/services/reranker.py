import logging
import math
from typing import List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        
    def _get_model(self):
        if self._model is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model
        
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
            
        model = self._get_model()
        pairs = [[query, text] for text in texts]
        
        # Predict returns logits for this model (e.g., -10 to +10)
        scores = model.predict(pairs)
        
        # Apply sigmoid to squash logits into a 0.0 to 1.0 range
        # This keeps it compatible with scoring.py which expects 0-1 scores
        return [1.0 / (1.0 + math.exp(-s)) for s in scores]
