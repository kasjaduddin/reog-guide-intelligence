"""
Unit tests for Embedding Service
"""

import os
import sys
import pytest

# src to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedding_service import EmbeddingService, get_embedding_service


@pytest.fixture
def embedding_service():
    """Get embedding service instance"""
    return get_embedding_service()


class TestEmbeddingService:
    """Test Embedding Service functionality"""
    
    def test_service_initialization(self, embedding_service):
        """Test that service initializes correctly"""
        assert embedding_service is not None
        assert embedding_service.model is not None
        assert embedding_service.collection is not None
    
    def test_embed_single_text(self, embedding_service):
        """Test single text embedding"""
        text = "Reog Ponorogo adalah kesenian tradisional"
        embedding = embedding_service.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM produces 384-dim vectors
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_batch(self, embedding_service):
        """Test batch embedding"""
        texts = [
            "Reog Ponorogo adalah kesenian tradisional",
            "Barongan adalah topeng raksasa",
            "Festival Reog diadakan setiap tahun"
        ]
        
        embeddings = embedding_service.embed_batch(texts, show_progress=False)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_search_similar_documents(self, embedding_service):
        """Test semantic search"""
        # Ensure KB is loaded
        embedding_service.load_knowledge_base()
        
        # Search
        query = "Apa itu Reog Ponorogo?"
        results = embedding_service.search(query, top_k=5)
        
        assert isinstance(results, list)
        assert all('content' in doc for doc in results)
        assert all('score' in doc for doc in results)
    
    def test_search_with_language_filter(self, embedding_service):
        """Test search with language filter"""
        embedding_service.load_knowledge_base()
        
        # Indonesian only
        results_id = embedding_service.search(
            "Apa itu barongan?",
            top_k=3,
            language_filter='id'
        )
        
        for doc in results_id:
            assert doc['metadata']['language'] == 'id'
        
        # English only
        results_en = embedding_service.search(
            "What is barongan?",
            top_k=3,
            language_filter='en'
        )
        
        for doc in results_en:
            assert doc['metadata']['language'] == 'en'
    
    def test_get_collection_stats(self, embedding_service):
        """Test collection statistics"""
        stats = embedding_service.get_collection_stats()
        
        assert 'total_documents' in stats
        assert 'categories' in stats
        assert 'languages' in stats
        
        if stats['total_documents'] > 0:
            assert len(stats['categories']) > 0
            assert len(stats['languages']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])