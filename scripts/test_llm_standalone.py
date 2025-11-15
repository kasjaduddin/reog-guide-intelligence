"""
Standalone LLM Testing Script
Tests Llama 3.2 with various prompts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_service import LLMService
from loguru import logger


def test_llm():
    """Test LLM with various scenarios"""
    
    logger.info("="*70)
    logger.info("LLM STANDALONE TESTING")
    logger.info("="*70)
    
    # Initialize
    llm = LLMService()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple question (ID)",
            "question": "Apa itu Reog Ponorogo?",
            "context": [
                {
                    'content': 'Reog Ponorogo adalah kesenian tradisional yang berasal dari Kabupaten Ponorogo, Jawa Timur. Pertunjukan ini terkenal dengan topeng barongan yang besar dan berat.',
                    'metadata': {'title': 'Pengenalan Reog', 'category': 'sejarah'}
                }
            ],
            "language": "id"
        },
        {
            "name": "Detailed question (ID)",
            "question": "Siapa yang menciptakan Reog Ponorogo?",
            "context": [
                {
                    'content': 'Menurut legenda, Reog Ponorogo diciptakan oleh Ki Ageng Kutu pada abad ke-15 sebagai bentuk kritik terhadap Raja Majapahit.',
                    'metadata': {'title': 'Sejarah Reog', 'category': 'sejarah'}
                }
            ],
            "language": "id"
        },
        {
            "name": "Simple question (EN)",
            "question": "What is Barongan?",
            "context": [
                {
                    'content': 'Barongan is the main icon of Reog Ponorogo, featuring a giant lion mask with a magnificent peacock feather crown weighing 50-60 kilograms.',
                    'metadata': {'title': 'Barongan Introduction', 'category': 'characters'}
                }
            ],
            "language": "en"
        },
        {
            "name": "No context available",
            "question": "Berapa harga tiket Festival Reog?",
            "context": [],
            "language": "id"
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {i}: {test_case['name']}")
        logger.info(f"{'='*70}")
        logger.info(f"Question: {test_case['question']}")
        logger.info(f"Language: {test_case['language']}")
        logger.info(f"Context docs: {len(test_case['context'])}")
        
        # Generate answer
        answer = llm.generate_answer(
            question=test_case['question'],
            context_documents=test_case['context'],
            language=test_case['language']
        )
        
        logger.info(f"\n✅ Answer:\n{answer}")
    
    # Test raw generation
    logger.info(f"\n{'='*70}")
    logger.info("TEST: Raw Generation (no context)")
    logger.info(f"{'='*70}")
    
    test_result = llm.test_generation("Jelaskan Reog Ponorogo dalam 2 kalimat.")
    logger.info(f"Prompt: {test_result['prompt']}")
    logger.info(f"Response: {test_result['response']}")
    
    logger.info(f"\n{'='*70}")
    logger.info("TESTING COMPLETE!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    test_llm()