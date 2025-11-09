"""
STT Performance Benchmarking
Measures latency, accuracy, and resource usage
"""

import sys
from pathlib import Path
import time
import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stt_service import STTService
from loguru import logger


def benchmark_stt():
    """Benchmark STT performance"""
    
    logger.info("="*70)
    logger.info("STT PERFORMANCE BENCHMARK")
    logger.info("="*70)
    
    # Initialize
    stt = STTService()
    
    # System info
    logger.info("\n💻 SYSTEM INFORMATION")
    logger.info("-"*70)
    logger.info(f"CPU: {psutil.cpu_count()} cores")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        logger.info("GPU: Not available (using CPU)")
    
    # Model info
    model_info = stt.get_model_info()
    logger.info(f"\nModel: {model_info['model_size']} ({model_info['parameters']})")
    logger.info(f"Device: {model_info['device']}")
    
    # Benchmark with Bahasa Indonesia and English audio files
    logger.info("\n⏱️ LATENCY BENCHMARK")
    logger.info("-"*70)
    
    test_cases = [
        ("Bahasa Indonesia - Short", "test_audio/bahasa_indonesia/short.wav"),
        ("Bahasa Indonesia - Medium", "test_audio/bahasa_indonesia/medium.wav"),
        ("Bahasa Indonesia - Long", "test_audio/bahasa_indonesia/long.wav"),
        ("English - Short", "test_audio/english/short.wav"),
        ("English - Medium", "test_audio/english/medium.wav"),
        ("English - Long", "test_audio/english/long.wav"),
    ]
    
    results = []
    
    for name, audio_path in test_cases:
        if not Path(audio_path).exists():
            logger.warning(f"⚠️ {audio_path} not found, skipping...")
            continue
        
        logger.info(f"\n📊 {name}")
        
        # Warm-up run
        stt.transcribe(audio_path)
        
        # Benchmark runs (3 iterations)
        latencies = []
        for i in range(3):
            start_time = time.time()
            result = stt.transcribe(audio_path)
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        logger.info(f"   Avg: {avg_latency:.2f}s")
        logger.info(f"   Min: {min_latency:.2f}s")
        logger.info(f"   Max: {max_latency:.2f}s")
        
        results.append({
            'name': name,
            'avg_latency': avg_latency,
            'min_latency': min_latency,
            'max_latency': max_latency
        })
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*70)
    
    if results:
        logger.info("\n📈 Results Summary:")
        for r in results:
            logger.info(f"   {r['name']}: {r['avg_latency']:.2f}s avg")


if __name__ == "__main__":
    benchmark_stt()