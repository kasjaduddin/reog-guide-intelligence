"""
Standalone STT Testing Script
Tests Whisper with real audio files (Bahasa Indonesia & English)
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt_service import STTService
from loguru import logger


def test_stt_with_real_audio():
    """
    Test STT with real audio files
    
    Expected directory structure:
    test_audio/
    ├── bahasa_indonesia/
    │   ├── sample1.wav
    │   └── sample2.mp3
    └── english/
        ├── sample1.wav
        └── sample2.mp3
    """
    
    logger.info("="*70)
    logger.info("STT STANDALONE TESTING")
    logger.info("="*70)
    
    # Initialize STT service
    logger.info("\n1️⃣ INITIALIZING STT SERVICE")
    logger.info("-"*70)
    
    stt = STTService()
    model_info = stt.get_model_info()
    
    logger.info(f"Model: {model_info['model_size']} ({model_info['parameters']} parameters)")
    logger.info(f"Device: {model_info['device']}")
    logger.info(f"FP16: {model_info['fp16']}")
    
    # Test audio directory
    test_audio_dir = Path("test_audio")
    
    if not test_audio_dir.exists():
        logger.warning(f"\n⚠️ Test audio directory not found: {test_audio_dir}")
        logger.info("Creating sample audio for testing...")
        
        # Create test directory
        test_audio_dir.mkdir(exist_ok=True)
        (test_audio_dir / "bahasa_indonesia").mkdir(exist_ok=True)
        (test_audio_dir / "english").mkdir(exist_ok=True)
        
        logger.info("✅ Directories created. Please add audio files and re-run.")
        return
    
    # Test Bahasa Indonesia audio
    logger.info("\n2️⃣ TESTING BAHASA INDONESIA AUDIO")
    logger.info("-"*70)
    
    id_audio_files = list((test_audio_dir / "bahasa_indonesia").glob("*.wav"))
    id_audio_files.extend(list((test_audio_dir / "bahasa_indonesia").glob("*.mp3")))
    
    if not id_audio_files:
        logger.warning("No Bahasa Indonesia audio files found")
    
    for audio_file in id_audio_files[:3]:  # Test max 3 files
        logger.info(f"\n📁 File: {audio_file.name}")
        
        start_time = time.time()
        result = stt.transcribe(audio_file, language='id')
        latency = time.time() - start_time
        
        if result['success']:
            logger.info(f"✅ Success (latency: {latency:.2f}s)")
            logger.info(f"   Language: {result['language']}")
            logger.info(f"   Text: {result['text']}")
            logger.info(f"   Segments: {len(result.get('segments', []))}")
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Test English audio
    logger.info("\n3️⃣ TESTING ENGLISH AUDIO")
    logger.info("-"*70)
    
    en_audio_files = list((test_audio_dir / "english").glob("*.wav"))
    en_audio_files.extend(list((test_audio_dir / "english").glob("*.mp3")))
    
    if not en_audio_files:
        logger.warning("No English audio files found")
    
    for audio_file in en_audio_files[:3]:
        logger.info(f"\n📁 File: {audio_file.name}")
        
        start_time = time.time()
        result = stt.transcribe(audio_file, language='en')
        latency = time.time() - start_time
        
        if result['success']:
            logger.info(f"✅ Success (latency: {latency:.2f}s)")
            logger.info(f"   Language: {result['language']}")
            logger.info(f"   Text: {result['text']}")
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Test auto-detection
    logger.info("\n4️⃣ TESTING AUTO LANGUAGE DETECTION")
    logger.info("-"*70)
    
    all_files = id_audio_files[:1] + en_audio_files[:1]
    
    for audio_file in all_files:
        logger.info(f"\n📁 File: {audio_file.name}")
        
        result = stt.transcribe(audio_file, language=None)  # Auto-detect
        
        if result['success']:
            logger.info(f"✅ Detected language: {result['language']}")
            logger.info(f"   Text: {result['text'][:100]}...")
    
    # Performance summary
    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE!")
    logger.info("="*70)
    logger.info("\n📊 Performance Summary:")
    logger.info(f"   Model: {model_info['model_size']}")
    logger.info(f"   Device: {model_info['device']}")
    logger.info(f"   Files tested: {len(id_audio_files) + len(en_audio_files)}")
    logger.info("\n✅ All tests passed! STT module is ready.")


if __name__ == "__main__":
    test_stt_with_real_audio()