import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

KAGGLE_USERNAME: str = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY: str = os.environ.get("KAGGLE_KEY", "")

RESULTS_PER_SOURCE: int = 20

RETRIEVER_TIMEOUT_SECONDS: int = 15

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

FAISS_TOP_K: int = 10

LLM_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"

LLM_MAX_NEW_TOKENS: int = 900

LLM_INPUT_COUNT: int = 5

DISPLAY_TOP_N: int = 5

LLM_MAX_RETRIES: int = 3
LLM_RETRY_BACKOFF_SECONDS: int = 5

def check_secrets() -> dict[str, bool]:
    """
    Returns a dict showing which optional secrets are configured.
    Used at startup to warn the user if a retrieval lane will be skipped.
    """
    return {
        "kaggle":  bool(KAGGLE_USERNAME and KAGGLE_KEY),
        "huggingface": True,   
        "datagov":     True,   
    }