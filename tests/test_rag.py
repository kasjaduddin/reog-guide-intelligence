"""
Unit tests for RAG Service
"""

import os
import sys
import pytest

# src to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_service import RAGService, get_rag_service


@pytest.fixture
def rag_service():
    """Get RAG service instance"""
    return get_rag_service()


class TestRAGService:
    """Test RAG Service functionality"""
    
    def test_service_initialization(self, rag_service):
        """Test that service initializes correctly"""
        assert rag_service is not None
        assert rag_service.embedding_service is not None
        assert rag_service.llm_service is not None
    
    def test_answer_question_indonesian(self, rag_service):
        """Test answering Indonesian question"""
        question = "Apa itu Reog Ponorogo?"
        result = rag_service.answer_question(question, language='id')
        
        assert 'answer' in result
        assert 'language' in result
        assert result['language'] == 'id'
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
    
    def test_answer_question_english(self, rag_service):
        """Test answering English question"""
        question = "What is Reog Ponorogo?"
        result = rag_service.answer_question(question, language='en')
        
        assert 'answer' in result
        assert result['language'] == 'en'
        assert len(result['answer']) > 0
    
    def test_answer_with_sources(self, rag_service):
        """Test that sources are returned"""
        question = "Siapa Ki Ageng Kutu?"
        result = rag_service.answer_question(
            question,
            language='id',
            return_sources=True
        )
        
        assert 'sources' in result
        assert isinstance(result['sources'], list)
        
        if result['success'] and result['sources']:
            source = result['sources'][0]
            assert 'title' in source
            assert 'category' in source
            assert 'score' in source
    
    def test_answer_with_timing(self, rag_service):
        """Test timing information"""
        question = "Apa itu barongan?"
        result = rag_service.answer_question(
            question,
            language='id',
            return_timing=True
        )
        
        if 'timing' in result:
            assert 'retrieval' in result['timing']
            assert 'generation' in result['timing']
            assert 'total' in result['timing']
    
    def test_language_autodetection(self, rag_service):
        """Test automatic language detection"""
        # Indonesian question
        result_id = rag_service.answer_question("Apa itu Reog?")
        assert result_id['language'] in ['id', 'en']
        
        # English question
        result_en = rag_service.answer_question("What is Reog?")
        assert result_en['language'] in ['id', 'en']
    
    def test_empty_question(self, rag_service):
        """Test handling of empty question"""
        result = rag_service.answer_question("", language='id')
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_irrelevant_question(self, rag_service):
        """Test handling of completely irrelevant question"""
        question = "What is quantum physics?"
        result = rag_service.answer_question(question, language='en')
        
        # Should return a polite "no information" response
        assert 'answer' in result
        assert len(result['answer']) > 0
    
    def test_get_stats(self, rag_service):
        """Test statistics retrieval"""
        stats = rag_service.get_stats()
        
        assert 'knowledge_base' in stats
        assert 'retrieval_config' in stats
        assert 'llm_config' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])