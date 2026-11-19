"""Configuration module for Research Intelligence & Evidence subsystem."""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "Autonomous Research Intelligence & Evidence Subsystem"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = "sqlite:///./research_intelligence.db"
    ECHO_SQL: bool = False

    # Vector Store (Qdrant)
    QDRANT_LOCATION: str = "./qdrant_data"  # local path or URL
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "evidence_chunks"
    VECTOR_DIMENSION: int = 384  # Standard MiniLM dimension
    CHECKPOINT_DATABASE_PATH: str = "./research_checkpoints.db"

    # Embeddings
    EMBEDDING_PROVIDER: str = "hash"  # 'hash' (<250MB RAM, recommended for Render free tier), 'fastembed', 'auto'
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # Web Search
    TAVILY_API_KEY: Optional[str] = None
    SEARCH_PROVIDER_ORDER: List[str] = ["tavily", "duckduckgo"]
    SEARCH_MAX_RESULTS: int = 5
    SEARCH_TIMEOUT_SECONDS: float = 10.0

    # Document Reader
    DOC_READER_TIMEOUT_SECONDS: float = 12.0
    DOC_READER_MAX_BYTES: int = 15 * 1024 * 1024  # 15 MB
    DOC_READER_USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 "
        "(Autonomous Research Agent)"
    )

    # Chunking
    DEFAULT_CHUNK_SIZE: int = 800  # characters
    DEFAULT_CHUNK_OVERLAP: int = 150  # characters

    # LLM Settings for Extraction & Credibility
    LLM_PROVIDER: str = "auto"  # 'auto', 'nvidia', 'gemini', 'groq', 'openai'
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_MODEL: str = "nvidia/nemotron-3.5-bolt-e4b"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT_SECONDS: float = 20.0
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BASE_DELAY_SECONDS: float = 0.5
    LLM_MAX_RPM: int = 60
    EXTRACTION_USE_LLM: bool = False
    EXTRACTION_LLM_MAX_CHUNKS: int = 3
    EXTRACTION_MAX_CONCURRENCY: int = 4
    EMBEDDING_CACHE_SIZE: int = 4096
    MAX_REPORT_CITATIONS: int = 50
    CORS_ALLOWED_ORIGINS: List[str] = ["*"]

    # Security
    BLOCKED_IP_NETWORKS: List[str] = [
        "127.0.0.0/8",      # Localhost
        "10.0.0.0/8",       # Private RFC 1918
        "172.16.0.0/12",    # Private RFC 1918
        "192.168.0.0/16",   # Private RFC 1918
        "169.254.0.0/16",   # Link-local / Cloud metadata (AWS/GCP/Azure)
        "0.0.0.0/8",        # Current network
        "::1/128",          # IPv6 Localhost
        "fc00::/7",         # IPv6 Private
        "fe80::/10"         # IPv6 Link-local
    ]
    ALLOWED_URL_SCHEMES: List[str] = ["http", "https"]


settings = Settings()
