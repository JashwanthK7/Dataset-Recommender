import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32).reshape(1, -1)

    def embed_datasets(self, datasets: list[dict]) -> np.ndarray:
        if not datasets:
            return np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())
        
        texts = [
            f"{ds.get('name', '')}. {ds.get('description', '')}"
            for ds in datasets
        ]
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)