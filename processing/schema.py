from typing import TypedDict

class NormalizedDataset(TypedDict):
    source: str
    name: str
    description: str
    url: str
    license: str
    last_updated: str
    format: str
    size_estimate: str