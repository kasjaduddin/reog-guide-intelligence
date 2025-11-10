"""
Embedding Service using Sentence-BERT and ChromaDB
Handles text embedding and vector database operations
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import config


class EmbeddingService:
    """
    Embedding and Vector Database Service
    
    Features:
    - Sentence-BERT multilingual embeddings
    - ChromaDB vector database
    - Semantic similarity search
    - Bilingual support (ID + EN)
    """
    
    def __init__(self):
        logger.info("📚 Initializing Embedding Service...")
        
        # 1. Load Sentence-BERT model
        logger.info(f"   Loading model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE
        )
        logger.info(f"   ✅ Model loaded on {config.EMBEDDING_DEVICE}")
        
        # 2. Initialize ChromaDB
        logger.info(f"   Initializing ChromaDB: {config.CHROMA_DB_PATH}")
        config.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_DB_PATH)
        )
        
        # 3. Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            logger.info(f"   ✅ Loaded existing collection: {config.CHROMA_COLLECTION_NAME}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Reog Ponorogo knowledge base",
                    "embedding_model": config.EMBEDDING_MODEL
                }
            )
            logger.info(f"   ✅ Created new collection: {config.CHROMA_COLLECTION_NAME}")
        
        logger.info("✅ Embedding Service ready!")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
        
        Returns:
            384-dimensional embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of texts
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings.tolist()
    
    def load_knowledge_base(self, force_reload: bool = False):
        """
        Load knowledge base from JSON into ChromaDB
        
        Args:
            force_reload: If True, clear existing data and reload
        """
        # Check if already loaded
        existing_count = self.collection.count()
        
        if existing_count > 0 and not force_reload:
            logger.info(f"📚 Knowledge base already loaded ({existing_count} documents)")
            return
        
        if force_reload and existing_count > 0:
            logger.warning("⚠️ Force reload: Deleting existing data...")
            self.chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={
                    "description": "Reog Ponorogo knowledge base",
                    "embedding_model": config.EMBEDDING_MODEL
                }
            )
        
        # Load knowledge base JSON
        logger.info(f"📂 Loading knowledge base from: {config.KNOWLEDGE_BASE_FILE}")
        
        if not config.KNOWLEDGE_BASE_FILE.exists():
            logger.error(f"❌ Knowledge base file not found: {config.KNOWLEDGE_BASE_FILE}")
            logger.info("Run: python scripts/prepare_knowledge_base.py")
            return
        
        with open(config.KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data['documents']
        logger.info(f"📄 Found {len(documents)} documents")
        
        # Prepare data for ChromaDB
        ids = [doc['id'] for doc in documents]
        texts = [doc['content'] for doc in documents]
        metadatas = [
            {
                'title': doc['title'],
                'category': doc['category'],
                'language': doc['language'],
                'keywords': ','.join(doc['keywords']),
                'word_count': str(doc['metadata'].get('word_count', 0)),
                'source_file': doc['metadata'].get('source_file', '')
            }
            for doc in documents
        ]
        
        # Generate embeddings (batch processing)
        logger.info("🔄 Generating embeddings...")
        embeddings = self.embed_batch(texts, show_progress=True)
        
        # Add to ChromaDB
        logger.info("💾 Adding to vector database...")
        
        # ChromaDB has a batch size limit, so we chunk the data
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            logger.info(f"   Added batch {i//batch_size + 1} ({end_idx}/{len(documents)})")
        
        logger.info(f"✅ Loaded {len(documents)} documents into ChromaDB")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        language_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search for relevant documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            language_filter: Filter by language ('id' or 'en')
            category_filter: Filter by category
        
        Returns:
            List of documents with scores
        """
        if top_k is None:
            top_k = config.RAG_TOP_K
        
        # Build where filter
        where_filter = {}
        if language_filter:
            where_filter['language'] = language_filter
        if category_filter:
            where_filter['category'] = category_filter
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        # Format results
        documents = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Convert distance to similarity score (1 - distance)
                # ChromaDB uses L2 distance by default
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)  # Convert to 0-1 range
                
                doc = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'score': similarity,
                    'distance': distance,
                    'metadata': results['metadatas'][0][i]
                }
                
                # Filter by score threshold
                if doc['score'] >= config.RAG_SCORE_THRESHOLD:
                    documents.append(doc)
        
        logger.info(f"🔍 Search: '{query[:50]}...' → {len(documents)} results")
        
        return documents
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        if count == 0:
            return {
                'total_documents': 0,
                'categories': [],
                'languages': []
            }
        
        # Sample items to get categories/languages
        sample_size = min(count, 100)
        sample = self.collection.peek(limit=sample_size)
        
        categories = set()
        languages = set()
        
        if sample['metadatas']:
            for meta in sample['metadatas']:
                categories.add(meta.get('category', 'unknown'))
                languages.add(meta.get('language', 'unknown'))
        
        return {
            'total_documents': count,
            'categories': sorted(list(categories)),
            'languages': sorted(list(languages)),
            'embedding_model': config.EMBEDDING_MODEL,
            'collection_name': config.CHROMA_COLLECTION_NAME
        }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by ID"""
        result = self.collection.get(ids=[doc_id])
        
        if result['ids']:
            return {
                'id': result['ids'][0],
                'content': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None


# Singleton instance
_embedding_service_instance = None

def get_embedding_service() -> EmbeddingService:
    """Get singleton Embedding service instance"""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance

# For backward compatibility
embedding_service = get_embedding_service()