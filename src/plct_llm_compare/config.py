"""Configuration settings for the plct_llm_compare module."""

import os

PLCT_AI_CTX_URL = "https://petljamediastorage.blob.core.windows.net/temp/ai-context-0.3.1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLLM_URL = "http://localhost:8000/v1"
