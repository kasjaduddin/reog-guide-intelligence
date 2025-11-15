"""
Configuration for AI Backend
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    
    # ========================================================================
    # WHISPER STT CONFIGURATION
    # ========================================================================
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE = "cuda" if (
        os.getenv("USE_GPU", "true").lower() == "true" and 
        os.getenv("WHISPER_USE_GPU", "true").lower() == "true"
    ) else "cpu"
    
    WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
    
    STT_LANGUAGE_HINT = None
    STT_TASK = "transcribe"
    
    # ========================================================================
    # EMBEDDING & VECTOR DB CONFIGURATION
    # ========================================================================
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE = "cuda" if (
        os.getenv("USE_GPU", "true").lower() == "true" and
        os.getenv("EMBEDDING_USE_GPU", "true").lower() == "true"
    ) else "cpu"
    
    # ChromaDB
    CHROMA_DB_PATH = DATA_DIR / "embeddings" / "chroma_db"
    CHROMA_COLLECTION_NAME = "reog_knowledge"
    
    # Knowledge Base
    KNOWLEDGE_BASE_FILE = DATA_DIR / "processed" / "knowledge_base.json"
    
    # RAG Settings
    RAG_TOP_K = 3  # Number of documents to retrieve
    RAG_SCORE_THRESHOLD = 0.12  # Minimum similarity score
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ========================================================================
    # LLM CONFIGURATION (Ollama)
    # ========================================================================
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
    
    # LLM Options
    LLM_TIMEOUT = 60  # seconds

config = Config()