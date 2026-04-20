import numpy as np
from deepface import DeepFace
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Production-ready Face Recognition using DeepFace + FaceNet512 (512-dim).
    Generates embeddings and supports averaging multiple photos.
    """

    def __init__(self):
        """Pre-load the model once at startup."""
        try:
            # Warm-up call to load FaceNet512
            _ = DeepFace.represent(img_path=np.zeros((160, 160, 3), dtype=np.uint8),
                                   model_name="Facenet512",
                                   enforce_detection=False)
            print("✅ FaceNet512 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load FaceNet512: {e}")
            raise

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate 512-dim embedding from a cropped face (160x160 RGB).
        Returns None if face is too blurry/low quality.
        """
        try:
            # DeepFace expects RGB
            if len(face_crop.shape) == 2 or face_crop.shape[2] == 1:
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)

            embedding_obj = DeepFace.represent(
                img_path=face_crop,
                model_name="Facenet512",
                enforce_detection=False,
                align=True,
                detector_backend="mtcnn"  # we already have MTCNN
            )

            embedding = np.array(embedding_obj[0]["embedding"], dtype=np.float32)
            return embedding

        except Exception as e:
            logger.debug(f"Embedding generation failed: {e}")
            return None

    def average_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Average multiple embeddings into one representative vector (used in registration)."""
        if not embeddings:
            raise ValueError("No embeddings to average")
        return np.mean(embeddings, axis=0)


# Singleton instance (used by similarity.py and later steps)
recognizer = FaceRecognizer()