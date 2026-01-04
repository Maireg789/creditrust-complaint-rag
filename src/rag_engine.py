# src/rag_engine.py
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, RateLimitError

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreditRAG")

# --- CODE 3: ROBUST ERROR HANDLING ---

class GenerationError(Exception):
    """Custom error for when the LLM fails"""
    pass

@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)), 
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def generate_answer_safe(chain, query):
    """
    Safely invokes the LLM chain with auto-retries on failure.
    """
    try:
        logger.info(f"Processing query: {query}")
        result = chain.invoke(query)
        return result
    except Exception as e:
        logger.error(f"LLM Generation failed: {e}")
        raise GenerationError(f"System is busy, please try again. Error: {e}")