import faiss
import numpy as np

class FAISSIndex:
    def __init__(self):
        self.index = None
        self.last_scores = []

    def build(self, embeddings: np.ndarray):
        if embeddings.size == 0:
            return
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int) -> list[int]:
        if self.index is None or self.index.ntotal == 0:
            self.last_scores = []
            return []

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        self.last_scores = scores[0].tolist()
        return indices[0].tolist()