"""Configuration settings for the plct_llm_compare module."""

import logging
import os

logger = logging.getLogger(__name__)

PLCT_AI_CTX_URL = os.getenv(
    "PLCT_AI_CTX_URL",
    "https://petljamediastorage.blob.core.windows.net/temp/ai-context-0.3.1",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")

logger.info("Config PLCT_AI_CTX_URL=%s", PLCT_AI_CTX_URL)
logger.info("Config VLLM_URL=%s", VLLM_URL)
logger.info("Config OPENAI_API_KEY=%s", "set" if OPENAI_API_KEY else "not set")
