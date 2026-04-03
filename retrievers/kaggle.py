import asyncio
import os
from .base import BaseRetriever

class KaggleRetriever(BaseRetriever):
    def __init__(self, username: str, key: str, max_results: int = 20, timeout: int = 15):
        super().__init__(max_results, timeout)
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key
        
    async def fetch(self, query: str) -> list[dict]:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            datasets = await asyncio.to_thread(
                api.dataset_list, search=query, max_size=None, min_size=None
            )
            
            results = []
            for count, item in enumerate(datasets):
                if count >= self.max_results:
                    break
                results.append({
                    "source": "Kaggle",
                    "name": getattr(item, 'title', getattr(item, 'ref', '')),
                    "description": getattr(item, 'subtitle', ''),
                    "url": f"https://www.kaggle.com/{getattr(item, 'ref', '')}",
                    "license": getattr(item, 'licenseName', 'unknown'),
                    "last_updated": str(getattr(item, 'lastUpdated', '')),
                    "format": "csv/zip",
                    "size_estimate": getattr(item, 'size', 'unknown')
                })
            return results
        except Exception:
            return []