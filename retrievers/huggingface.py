import urllib.parse
from .base import BaseRetriever
from utils import async_fetch_json

class HuggingFaceRetriever(BaseRetriever):
    async def fetch(self, query: str) -> list[dict]:
        encoded_query = urllib.parse.quote(query)
        url = f"https://huggingface.co/api/datasets?search={encoded_query}&limit={self.max_results}&full=true"
        
        try:
            data = await async_fetch_json(url, timeout=self.timeout)
            results = []
            for item in data:
                results.append({
                    "source": "Hugging Face",
                    "name": item.get("id", ""),
                    "description": item.get("description", ""),
                    "url": f"https://huggingface.co/datasets/{item.get('id', '')}",
                    "license": item.get("cardData", {}).get("license", "unknown") if isinstance(item.get("cardData"), dict) else "unknown",
                    "last_updated": item.get("lastModified", ""),
                    "format": "parquet", 
                    "size_estimate": "unknown" 
                })
            return results
        except Exception:
            return []