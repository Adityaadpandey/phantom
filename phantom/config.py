"""
Central OpenAI / LLM configuration — single source of truth for the whole app.

Environment variables:
    HF_TOKEN       API key (Hugging Face or OpenAI-compatible)
    API_BASE_URL   LLM endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     Primary model  (default: gpt-5.4)
    FALLBACK_MODEL_NAME  Fallback model  (default: same as MODEL_NAME)
"""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY: str         = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL: str    = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME: str      = os.getenv("MODEL_NAME") or "gpt-5.4"
FALLBACK_MODEL: str  = os.getenv("FALLBACK_MODEL_NAME") or MODEL_NAME
