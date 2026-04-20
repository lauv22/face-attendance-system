import numpy as np
from typing import List, Tuple, Optional
from models.face_recognizer import recognizer
import logging

logger = logging.getLogger(__name__)

class SimilarityMatcher:
    """
    Fast cosine similarity matcher with in-memory caching.
    Compares one face embedding against ALL stored embeddings at once (vectorized).
    """

    def __init__(self):
        self.cached_embeddings: np.ndarray = None   # shape: (N, 512)
        self.cached_person_ids: List[int] = []      # corresponding person_id for each row

    def load_embeddings_from_db(self, embeddings_data: List[Tuple[int, np.ndarray]]):
        """Called once on app startup or after registration. Loads all embeddings into memory."""
        if not embeddings_data:
            self.cached_embeddings = None
            self.cached_person_ids = []
            return

        self.cached_person_ids = [pid for pid, emb in embeddings_data]
        self.cached_embeddings = np.array([emb for pid, emb in embeddings_data], dtype=np.float32)
        print(f"✅ Loaded {len(self.cached_person_ids)} face embeddings into memory cache")

    def find_match(self, query_embedding: np.ndarray, threshold: float = 0.65) -> Tuple[Optional[int], float]:
        """
        Compare query embedding with all cached embeddings using cosine similarity.
        Returns (person_id, similarity_score) or (None, score) if no match.
        """
        if self.cached_embeddings is None or len(self.cached_embeddings) == 0:
            return None, 0.0

        # Vectorized cosine similarity (very fast)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        cache_norm = self.cached_embeddings / np.linalg.norm(self.cached_embeddings, axis=1, keepdims=True)
        similarities = np.dot(cache_norm, query_norm)

        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])

        if best_score > threshold:
            return self.cached_person_ids[best_idx], best_score

        return None, best_score


# Global instance (will be used by Flask app later)
matcher = SimilarityMatcher()