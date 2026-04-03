from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    def __init__(self, max_results: int = 20, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout

    @abstractmethod
    async def fetch(self, query: str) -> list[dict]:
        pass