"""
Unit tests for STT Service
"""

import pytest
import numpy as np
import soundfile as sf

# src to sys.path to import the module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt_service import get_stt_service


# Test fixtures
@pytest.fixture
def stt_service():
    """Get STT service instance"""
    return get_stt_service()


@pytest.fixture
def sample_audio_id(tmp_path):
    """Create sample Indonesian audio for testing"""
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    audio_path = tmp_path / "test_audio_id.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


@pytest.fixture
def sample_audio_en(tmp_path):
    """Create sample English audio for testing"""
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 880 * t) * 0.3
    
    audio_path = tmp_path / "test_audio_en.wav"
    sf.write(audio_path, audio, sample_rate)
    return audio_path


# Test cases
class TestSTTService:
    """Test STT Service functionality"""
    
    def test_service_initialization(self, stt_service):
        assert stt_service is not None
        assert stt_service.model is not None
        assert stt_service.model_size in ['tiny', 'base', 'small', 'medium', 'large']
    
    def test_get_model_info(self, stt_service):
        info = stt_service.get_model_info()
        assert 'model_size' in info
        assert 'device' in info
        assert 'fp16' in info
        assert 'parameters' in info
    
    def test_transcribe_audio_file(self, stt_service, sample_audio_id):
        result = stt_service.transcribe(sample_audio_id)
        assert result['success'] is True
        assert 'text' in result
        assert 'language' in result
        assert isinstance(result['text'], str)
    
    def test_transcribe_with_language_hint(self, stt_service, sample_audio_id):
        result = stt_service.transcribe(sample_audio_id, language='id')
        assert result['success'] is True
    
    def test_transcribe_nonexistent_file(self, stt_service):
        result = stt_service.transcribe("nonexistent_file.wav")
        assert result['success'] is False
        assert 'error' in result
    
    def test_transcribe_bytes(self, stt_service, sample_audio_id):
        with open(sample_audio_id, 'rb') as f:
            audio_bytes = f.read()
        result = stt_service.transcribe_bytes(audio_bytes)
        assert result['success'] is True
        assert 'text' in result
    
    def test_transcribe_with_timestamps(self, stt_service, sample_audio_id):
        result = stt_service.transcribe_with_timestamps(sample_audio_id)
        assert result['success'] is True
        assert 'words' in result or 'segments' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])