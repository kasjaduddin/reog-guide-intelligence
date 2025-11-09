"""
Speech-to-Text Service using OpenAI Whisper
Handles audio transcription with automatic language detection
"""

import whisper
import torch
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Optional, Union
from loguru import logger

from src.config import config
from src.text_normalizer import TextNormalizer  # <-- import normalizer


class STTService:
    """
    Speech-to-Text service using Whisper model
    
    Features:
    - Automatic language detection (ID/EN)
    - GPU acceleration (if available)
    - Multiple audio format support
    - Robust error handling
    - Text normalization for cultural/local terms
    """
    
    def __init__(self, model_size: str = None):
        self.model_size = model_size or config.WHISPER_MODEL
        self.device = config.WHISPER_DEVICE
        
        logger.info(f"🎤 Initializing Whisper STT...")
        logger.info(f"   Model: {self.model_size}")
        logger.info(f"   Device: {self.device}")
        
        # Load Whisper model
        self.model = whisper.load_model(
            self.model_size,
            device=self.device
        )
        
        # Check if FP16 available (for GPU)
        self.fp16 = (self.device == 'cuda' and torch.cuda.is_available())
        
        # Initialize text normalizer
        self.normalizer = TextNormalizer()
        
        logger.info(f"✅ Whisper STT loaded successfully")
        logger.info(f"   FP16: {self.fp16}")
    
    def transcribe(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = 'transcribe'
    ) -> Dict:
        try:
            audio_path = str(audio_path)
            
            logger.info(f"🎤 Transcribing: {Path(audio_path).name}")
            if language:
                logger.info(f"   Language hint: {language}")
            
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                fp16=self.fp16,
                verbose=False,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            text = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            segments = result.get('segments', [])
            
            # Apply text normalization
            normalized_text = self.normalizer.normalize(text)
            
            logger.info(f"✅ Transcription complete:")
            logger.info(f"   Language: {detected_language}")
            logger.info(f"   Length: {len(normalized_text)} chars")
            logger.info(f"   Text: '{normalized_text[:100]}{'...' if len(normalized_text) > 100 else ''}'")
            
            return {
                'text': normalized_text,
                'language': detected_language,
                'segments': segments,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ STT Error: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'error': str(e),
                'success': False
            }
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> Dict:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            result = self.transcribe(tmp_path, language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        return result
    
    def transcribe_with_timestamps(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        result = self.transcribe(audio_path, language)
        
        if not result['success']:
            return result
        
        words = []
        for segment in result['segments']:
            segment_start = segment.get('start', 0)
            segment_text = segment.get('text', '').strip()
            
            segment_words = segment_text.split()
            duration = segment.get('end', segment_start) - segment_start
            word_duration = duration / max(len(segment_words), 1)
            
            for i, word in enumerate(segment_words):
                words.append({
                    'word': word,
                    'start': segment_start + (i * word_duration),
                    'end': segment_start + ((i + 1) * word_duration)
                })
        
        result['words'] = words
        return result
    
    def get_model_info(self) -> Dict:
        return {
            'model_size': self.model_size,
            'device': self.device,
            'fp16': self.fp16,
            'parameters': {
                'tiny': '39M',
                'base': '74M',
                'small': '244M',
                'medium': '769M',
                'large': '1550M'
            }.get(self.model_size, 'unknown')
        }


_stt_service_instance = None

def get_stt_service() -> STTService:
    global _stt_service_instance
    if _stt_service_instance is None:
        _stt_service_instance = STTService()
    return _stt_service_instance

stt_service = get_stt_service()