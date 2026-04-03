import aiohttp
import logging
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def async_fetch_json(
    url: str, 
    headers: dict[str, str] | None = None, 
    params: dict[str, Any] | None = None, 
    timeout: int = 15
) -> Any:
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Fetch failed for {url}: {e}")
            raise