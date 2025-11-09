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
    
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
    WHISPER_DEVICE = "cuda" if (
        os.getenv("USE_GPU", "true").lower() == "true" and 
        os.getenv("WHISPER_USE_GPU", "true").lower() == "true"
    ) else "cpu"
    
    WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
    
    STT_LANGUAGE_HINT = None
    STT_TASK = "transcribe"
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()