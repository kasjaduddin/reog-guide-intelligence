"""
RAG Quality Evaluation Framework
Evaluates retrieval accuracy and answer quality
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_service import RAGService
from loguru import logger


class RAGEvaluator:
    """Evaluate RAG system quality"""
    
    def __init__(self):
        self.rag = RAGService()
    
    def evaluate_retrieval(self, test_questions_file: str):
        """
        Evaluate retrieval accuracy
        
        Test questions file format (JSON):
        [
            {
                "question": "Apa itu Reog?",
                "language": "id",
                "relevant_doc_ids": ["doc_001", "doc_002"]
            },
            ...
        ]
        """
        logger.info("📊 EVALUATING RETRIEVAL ACCURACY")
        logger.info("-"*70)
        
        # Load test questions
        with open(test_questions_file, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        results = []
        
        for test_q in test_questions:
            question = test_q['question']
            relevant_ids = set(test_q['relevant_doc_ids'])
            
            # Retrieve
            retrieved_docs = self.rag.embedding_service.search(
                query=question,
                top_k=5,
                language_filter=test_q.get('language')
            )
            
            retrieved_ids = {doc['id'] for doc in retrieved_docs}
            
            # Calculate metrics
            true_positives = len(relevant_ids & retrieved_ids)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            recall = true_positives / len(relevant_ids) if relevant_ids else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'question': question,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"   P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
        
        # Average metrics
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        
        logger.info("\n📈 AVERAGE METRICS:")
        logger.info(f"   Precision: {avg_precision:.2f}")
        logger.info(f"   Recall: {avg_recall:.2f}")
        logger.info(f"   F1 Score: {avg_f1:.2f}")
        
        return results
    
    def evaluate_answer_quality(self, test_qa_pairs_file: str):
        """
        Evaluate answer quality
        
        Test QA pairs file format (JSON):
        [
            {
                "question": "Apa itu Reog?",
                "expected_answer_keywords": ["reog", "ponorogo", "tradisional"],
                "language": "id"
            },
            ...
        ]
        """
        logger.info("\n📊 EVALUATING ANSWER QUALITY")
        logger.info("-"*70)
        
        with open(test_qa_pairs_file, 'r', encoding='utf-8') as f:
            test_pairs = json.load(f)
        
        results = []
        
        for pair in test_pairs:
            question = pair['question']
            expected_kw = pair['expected_answer_keywords']
            
            # Get answer
            result = self.rag.answer_question(
                question=question,
                language=pair.get('language', 'id')
            )
            
            answer = result.get('answer', '').lower()
            
            # Calculate keyword coverage
            keywords_found = [kw for kw in expected_kw if kw.lower() in answer]
            coverage = len(keywords_found) / len(expected_kw)
            
            results.append({
                'question': question,
                'coverage': coverage,
                'success': result.get('success', False)
            })
            
            logger.info(f"Q: {question[:50]}...")
            logger.info(f"   Coverage: {coverage:.1%}, Success: {result.get('success')}")
        
        # Average metrics
        avg_coverage = sum(r['coverage'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        logger.info("\n📈 AVERAGE METRICS:")
        logger.info(f"   Keyword Coverage: {avg_coverage:.1%}")
        logger.info(f"   Success Rate: {success_rate:.1%}")
        
        return results


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # Note: Create test files first
    # evaluator.evaluate_retrieval("data/test/retrieval_test.json")
    # evaluator.evaluate_answer_quality("data/test/qa_test.json")
    
    logger.info("To run evaluation, create test JSON files first.")