import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from feature_extractor import FeatureExtractor

INDEX_PATH = Path(__file__).parent / "index.pkl"
PRODUCTS_DIR = Path(__file__).parent / "products"


class Recommender:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Índice não encontrado: {INDEX_PATH}. Execute indexer.py primeiro."
            )
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.embeddings = data["embeddings"]
        self.extractor = FeatureExtractor()

    def recommend(self, query_image: Image.Image, top_n: int = 5) -> list[dict]:
        query_emb = self.extractor.extract(query_image)
        similarities = self.embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_n]
        results = []
        for idx in top_indices:
            results.append({
                "name": self.names[idx],
                "path": str(PRODUCTS_DIR / self.names[idx]),
                "score": float(similarities[idx]),
            })
        return results
