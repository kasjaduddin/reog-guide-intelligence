"""
RAG (Retrieval-Augmented Generation) Service
Orchestrates the complete Q&A pipeline
"""

from typing import Dict, List, Optional
import time
from loguru import logger

from src.embedding_service import get_embedding_service
from src.llm_service import get_llm_service
from src.config import config


class RAGService:
    """
    RAG Pipeline Service
    
    Complete workflow:
    1. Embed user question
    2. Retrieve relevant documents from vector DB
    3. Generate answer using LLM with context
    4. Format and return result with sources
    
    Features:
    - Bilingual support (ID + EN)
    - Source attribution
    - Quality filtering
    - Performance tracking
    """
    
    def __init__(self):
        logger.info("🔗 Initializing RAG Service...")
        
        self.embedding_service = get_embedding_service()
        self.llm_service = get_llm_service()
        
        # Load knowledge base if not already loaded
        self.embedding_service.load_knowledge_base()
        
        logger.info("✅ RAG Service ready!")
    
    def answer_question(
        self,
        question: str,
        language: Optional[str] = None,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        return_timing: bool = False
    ) -> Dict:
        """
        Complete RAG pipeline: Answer question with retrieved context
        
        Args:
            question: User question
            language: Target language ('id', 'en', or None for auto)
            top_k: Number of documents to retrieve (default: from config)
            return_sources: Include source documents in response
            return_timing: Include timing information
        
        Returns:
            dict with:
                - answer: Generated answer text
                - sources: List of source documents (if return_sources=True)
                - language: Detected/used language
                - success: True if successful
                - timing: Dict with timing info (if return_timing=True)
                - error: Error message (if failed)
        """
        timing = {}
        start_time = time.time()
        
        try:
            # Validate input
            if not question or not question.strip():
                return self._error_response("Empty question", language or 'id')
            
            question = question.strip()
            logger.info(f"🔍 RAG Pipeline: '{question[:60]}...'")
            
            # Auto-detect language if not specified
            if language is None:
                language = self._detect_language(question)
                logger.info(f"   Auto-detected language: {language}")
            
            # STEP 1: Retrieve relevant documents
            retrieval_start = time.time()
            
            documents = self.embedding_service.search(
                query=question,
                top_k=top_k or config.RAG_TOP_K,
                language_filter=language
            )
            
            timing['retrieval'] = time.time() - retrieval_start
            
            if not documents:
                logger.warning("   ⚠️ No relevant documents found")
                return self._no_info_response(question, language)
            
            logger.info(f"   ✅ Retrieved {len(documents)} documents")
            for i, doc in enumerate(documents, 1):
                logger.debug(f"      {i}. {doc['metadata'].get('title', 'Unknown')} (score: {doc['score']:.3f})")
            
            # STEP 2: Generate answer using LLM
            generation_start = time.time()
            
            answer = self.llm_service.generate_answer(
                question=question,
                context_documents=documents,
                language=language
            )
            
            timing['generation'] = time.time() - generation_start
            
            if not answer:
                logger.error("   ❌ LLM generation failed")
                return self._error_response("Answer generation failed", language)
            
            logger.info(f"   ✅ Generated answer: {len(answer)} chars")
            logger.debug(f"      Answer: {answer[:100]}...")
            
            # STEP 3: Format sources
            sources = []
            if return_sources:
                sources = self._format_sources(documents)
            
            # Calculate total time
            timing['total'] = time.time() - start_time
            
            logger.info(f"   ⏱️ Total time: {timing['total']:.2f}s")
            
            # Build response
            response = {
                'answer': answer,
                'language': language,
                'success': True
            }
            
            if return_sources:
                response['sources'] = sources
            
            if return_timing:
                response['timing'] = timing
            
            return response
            
        except Exception as e:
            logger.error(f"❌ RAG Pipeline Error: {e}")
            return {
                'answer': self._get_error_message(language or 'id'),
                'sources': [],
                'language': language or 'id',
                'error': str(e),
                'success': False
            }
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on keywords
        (For production, use langdetect library)
        """
        # Common Indonesian words
        id_keywords = ['apa', 'yang', 'adalah', 'ini', 'itu', 'dengan', 'dari', 'ke', 'di', 'untuk']
        # Common English words
        en_keywords = ['what', 'is', 'the', 'this', 'that', 'with', 'from', 'to', 'in', 'for']
        
        text_lower = text.lower()
        
        id_score = sum(1 for kw in id_keywords if kw in text_lower)
        en_score = sum(1 for kw in en_keywords if kw in text_lower)
        
        return 'id' if id_score >= en_score else 'en'
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """Format source documents for response"""
        sources = []
        
        for doc in documents:
            source = {
                'title': doc['metadata'].get('title', 'Unknown'),
                'category': doc['metadata'].get('category', 'Unknown'),
                'score': round(doc['score'], 3),
                'excerpt': doc['content'][:150] + '...' if len(doc['content']) > 150 else doc['content']
            }
            sources.append(source)
        
        return sources
    
    def _error_response(self, error_msg: str, language: str) -> Dict:
        """Generate error response"""
        return {
            'answer': self._get_error_message(language),
            'sources': [],
            'language': language,
            'error': error_msg,
            'success': False
        }
    
    def _no_info_response(self, question: str, language: str) -> Dict:
        """Generate 'no information' response"""
        if language == 'id':
            answer = "Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan Anda di basis pengetahuan saya tentang Reog Ponorogo."
        else:
            answer = "I'm sorry, I couldn't find relevant information to answer your question in my knowledge base about Reog Ponorogo."
        
        return {
            'answer': answer,
            'sources': [],
            'language': language,
            'success': False
        }
    
    def _get_error_message(self, language: str) -> str:
        """Get error message in specified language"""
        if language == 'id':
            return "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. Silakan coba lagi."
        else:
            return "I'm sorry, there was an error processing your question. Please try again."
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics"""
        kb_stats = self.embedding_service.get_collection_stats()
        
        return {
            'knowledge_base': kb_stats,
            'retrieval_config': {
                'top_k': config.RAG_TOP_K,
                'score_threshold': config.RAG_SCORE_THRESHOLD,
                'embedding_model': config.EMBEDDING_MODEL
            },
            'llm_config': {
                'model': config.OLLAMA_MODEL,
                'temperature': config.OLLAMA_TEMPERATURE,
                'max_tokens': config.OLLAMA_MAX_TOKENS
            }
        }


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get singleton RAG service instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance

# For backward compatibility
rag_service = get_rag_service()