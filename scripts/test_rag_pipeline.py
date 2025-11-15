"""
End-to-end RAG Pipeline Testing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_service import RAGService
from loguru import logger


def test_rag_pipeline():
    """Test RAG pipeline with various questions"""
    
    logger.info("="*70)
    logger.info("RAG PIPELINE END-TO-END TESTING")
    logger.info("="*70)
    
    # Initialize
    rag = RAGService()
    
    # Display stats
    logger.info("\n📊 SYSTEM STATISTICS")
    logger.info("-"*70)
    stats = rag.get_stats()
    
    kb = stats['knowledge_base']
    logger.info(f"Knowledge Base:")
    logger.info(f"  Total documents: {kb['total_documents']}")
    logger.info(f"  Categories: {', '.join(kb['categories'])}")
    logger.info(f"  Languages: {', '.join(kb['languages'])}")
    
    # Test questions
    test_cases = [
        {
            "question": "Apa itu Reog Ponorogo?",
            "language": "id",
            "expected_keywords": ["reog", "ponorogo", "tradisional", "jawa"]
        },
        {
            "question": "Siapa yang menciptakan Reog?",
            "language": "id",
            "expected_keywords": ["ki ageng kutu", "pencipta"]
        },
        {
            "question": "Apa itu barongan?",
            "language": "id",
            "expected_keywords": ["barongan", "topeng", "singa", "merak"]
        },
        {
            "question": "What is Reog Ponorogo?",
            "language": "en",
            "expected_keywords": ["reog", "ponorogo", "traditional", "art"]
        },
        {
            "question": "Who created Reog?",
            "language": "en",
            "expected_keywords": ["ki ageng kutu", "created"]
        },
        {
            "question": "When is the Reog festival held?",
            "language": "en",
            "expected_keywords": ["festival", "annual", "suro"]
        },
    ]
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST {i}/{len(test_cases)}")
        logger.info(f"{'='*70}")
        
        question = test_case['question']
        language = test_case['language']
        
        logger.info(f"Question: {question}")
        logger.info(f"Language: {language}")
        
        # Get answer
        result = rag.answer_question(
            question=question,
            language=language,
            return_sources=True,
            return_timing=True
        )
        
        # Display result
        logger.info(f"\nStatus: {'✅ Success' if result['success'] else '❌ Failed'}")
        logger.info(f"\nAnswer:\n{result['answer']}")
        
        if result.get('sources'):
            logger.info(f"\nSources ({len(result['sources'])}):")
            for j, source in enumerate(result['sources'], 1):
                logger.info(f"  {j}. {source['title']} (score: {source['score']})")
        
        if result.get('timing'):
            timing = result['timing']
            logger.info(f"\nTiming:")
            logger.info(f"  Retrieval: {timing['retrieval']:.3f}s")
            logger.info(f"  Generation: {timing['generation']:.3f}s")
            logger.info(f"  Total: {timing['total']:.3f}s")
        
        # Validate answer quality
        answer_lower = result['answer'].lower()
        keywords_found = [
            kw for kw in test_case.get('expected_keywords', [])
            if kw.lower() in answer_lower
        ]
        
        quality_score = len(keywords_found) / max(len(test_case.get('expected_keywords', [])), 1)
        
        logger.info(f"\nQuality Check:")
        logger.info(f"  Keywords found: {keywords_found}")
        logger.info(f"  Quality score: {quality_score:.1%}")
        
        results.append({
            'question': question,
            'success': result['success'],
            'quality_score': quality_score,
            'timing': result.get('timing', {}).get('total', 0)
        })
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*70}")
    
    success_rate = sum(1 for r in results if r['success']) / len(results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_latency = sum(r['timing'] for r in results) / len(results)
    
    logger.info(f"\n✅ Success rate: {success_rate:.1%}")
    logger.info(f"📊 Avg quality score: {avg_quality:.1%}")
    logger.info(f"⏱️ Avg latency: {avg_latency:.2f}s")
    
    logger.info("\nDetailed Results:")
    for i, r in enumerate(results, 1):
        status = "✅" if r['success'] else "❌"
        logger.info(f"  {i}. {status} {r['question'][:50]}... (Q: {r['quality_score']:.0%}, T: {r['timing']:.2f}s)")
    
    logger.info(f"\n{'='*70}")
    logger.info("TESTING COMPLETE!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    test_rag_pipeline()