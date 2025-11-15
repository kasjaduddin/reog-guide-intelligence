"""
Unit tests for LLM Service
"""

import os
import sys
import pytest

# src to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_service import LLMService, get_llm_service


@pytest.fixture
def llm_service():
    """Get LLM service instance"""
    return get_llm_service()


class TestLLMService:
    """Test LLM Service functionality"""
    
    def test_service_initialization(self, llm_service):
        """Test that service initializes correctly"""
        assert llm_service is not None
        assert llm_service.base_url is not None
        assert llm_service.model is not None
    
    def test_simple_generation(self, llm_service):
        """Test simple text generation"""
        prompt = "Sebutkan 3 kota di Indonesia."
        result = llm_service.generate(prompt, temperature=0.5, max_tokens=50)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generation_with_system_prompt(self, llm_service):
        """Test generation with system prompt"""
        system_prompt = "Anda adalah asisten yang selalu menjawab dalam 1 kalimat."
        prompt = "Apa ibukota Indonesia?"
        
        result = llm_service.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=30
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_answer_indonesian(self, llm_service):
        """Test answer generation in Indonesian"""
        question = "Apa itu Reog Ponorogo?"
        context_documents = [
            {
                'content': 'Reog Ponorogo adalah kesenian tradisional dari Ponorogo, Jawa Timur.',
                'metadata': {'title': 'Pengenalan Reog', 'category': 'sejarah'}
            }
        ]
        
        answer = llm_service.generate_answer(question, context_documents, language='id')
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert 'Reog' in answer or 'reog' in answer.lower()
    
    def test_generate_answer_english(self, llm_service):
        """Test answer generation in English"""
        question = "What is Reog Ponorogo?"
        context_documents = [
            {
                'content': 'Reog Ponorogo is a traditional art form from Ponorogo, East Java.',
                'metadata': {'title': 'Introduction to Reog', 'category': 'history'}
            }
        ]
        
        answer = llm_service.generate_answer(question, context_documents, language='en')
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_handle_empty_context(self, llm_service):
        """Test behavior with empty context"""
        question = "Apa itu Reog?"
        context_documents = []
        
        answer = llm_service.generate_answer(question, context_documents, language='id')
        
        # Should still return something (probably saying no info available)
        assert isinstance(answer, str)
    
    def test_test_generation_method(self, llm_service):
        """Test the test_generation utility method"""
        result = llm_service.test_generation()
        
        assert 'prompt' in result
        assert 'response' in result
        assert 'success' in result
        assert result['success'] == (len(result['response']) > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])