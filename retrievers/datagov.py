import urllib.parse
from .base import BaseRetriever
from utils import async_fetch_json

class DataGovRetriever(BaseRetriever):
    async def fetch(self, query: str) -> list[dict]:
        encoded_query = urllib.parse.quote(query)
        url = f"https://catalog.data.gov/api/3/action/package_search?q={encoded_query}&rows={self.max_results}"
        
        try:
            data = await async_fetch_json(url, timeout=self.timeout)
            results = []
            for item in data.get("result", {}).get("results", []):
                formats = [res.get("format", "") for res in item.get("resources", [])]
                fmt = formats[0] if formats else "unknown"
                
                results.append({
                    "source": "data.gov",
                    "name": item.get("title", ""),
                    "description": item.get("notes", ""),
                    "url": f"https://catalog.data.gov/dataset/{item.get('name', '')}",
                    "license": item.get("license_title", "unknown"),
                    "last_updated": item.get("metadata_modified", ""),
                    "format": fmt,
                    "size_estimate": "unknown"
                })
            return results
        except Exception:
            return []