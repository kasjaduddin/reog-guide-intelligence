"""
LLM Service using Ollama (Llama 3.2)
Handles answer generation from context and questions
"""

import requests
from typing import List, Dict, Optional
from loguru import logger

from src.config import config


class LLMService:
    """
    Large Language Model Service via Ollama
    
    Features:
    - Answer generation from context
    - Bilingual support (ID + EN)
    - Prompt engineering for RAG
    - Streaming support (optional)
    - Configurable parameters
    """
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        
        logger.info(f"🤖 Initializing LLM Service...")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Model: {self.model}")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m['name'] for m in models_data.get('models', [])]
                
                if self.model in available_models:
                    logger.info(f"   ✅ Connected to Ollama: {self.model}")
                else:
                    logger.warning(f"   ⚠️ Model {self.model} not found!")
                    logger.warning(f"   Available models: {available_models}")
                    logger.info(f"   Run: ollama pull {self.model}")
            else:
                logger.error(f"   ❌ Ollama connection failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Cannot connect to Ollama: {e}")
            logger.error("   Make sure Ollama is running: ollama serve")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text completion
        
        Args:
            prompt: User prompt
            system_prompt: System instruction (prepended to prompt)
            temperature: Sampling temperature (0-1, default from config)
            max_tokens: Maximum tokens to generate (default from config)
            stream: Stream response (not implemented yet)
        
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = config.OLLAMA_TEMPERATURE
        
        if max_tokens is None:
            max_tokens = config.OLLAMA_MAX_TOKENS
        
        # Build full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": stream,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=60  # 60 seconds timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                logger.info(f"🤖 LLM generated {len(generated_text)} characters")
                logger.debug(f"   Text: {generated_text[:100]}...")
                
                return generated_text
            else:
                logger.error(f"❌ LLM generation failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error("❌ LLM request timeout (> 60s)")
            return ""
        except Exception as e:
            logger.error(f"❌ LLM Error: {e}")
            return ""
    
    def generate_answer(
        self,
        question: str,
        context_documents: List[Dict],
        language: str = 'id'
    ) -> str:
        """
        Generate answer based on question and retrieved context
        
        This is the main method for RAG pipeline
        
        Args:
            question: User question
            context_documents: Retrieved documents from vector search
            language: Response language ('id' or 'en')
        
        Returns:
            Generated answer
        """
        # Build context from documents
        context = self._build_context(context_documents)
        
        # Get system prompt
        system_prompt = self._get_system_prompt(language)
        
        # Build user prompt
        user_prompt = self._build_user_prompt(question, context, language)
        
        # Generate answer
        answer = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=config.OLLAMA_TEMPERATURE,
            max_tokens=256  # Shorter for concise answers
        )
        
        # Post-process answer
        answer = self._postprocess_answer(answer, language)
        
        return answer
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            title = doc['metadata'].get('title', 'Unknown')
            category = doc['metadata'].get('category', '')
            
            context_parts.append(f"[Dokumen {i}: {title} ({category})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt based on language"""
        if language == 'id':
            return """Anda adalah pemandu virtual museum Reog Ponorogo yang ramah dan berpengetahuan luas.

TUGAS ANDA:
- Menjawab pertanyaan pengunjung tentang Reog Ponorogo
- Gunakan HANYA informasi dari konteks yang diberikan
- Berikan jawaban yang informatif namun ringkas (2-4 kalimat)
- Bersikap ramah dan antusias tentang budaya Reog

ATURAN PENTING:
1. Jika informasi tidak ada dalam konteks, katakan "Maaf, saya tidak memiliki informasi tentang hal tersebut dalam basis pengetahuan saya."
2. Jangan mengarang atau menambahkan informasi yang tidak ada di konteks
3. Gunakan Bahasa Indonesia yang jelas dan mudah dipahami
4. Fokus pada informasi yang paling relevan dengan pertanyaan
5. Jangan memulai jawaban dengan sapaan pembuka"""
        
        else:  # English
            return """You are a friendly and knowledgeable Reog Ponorogo virtual museum guide.

YOUR TASK:
- Answer visitors' questions about Reog Ponorogo
- Use ONLY information from the provided context
- Provide informative yet concise answers (2-4 sentences)
- Be friendly and enthusiastic about Reog culture

IMPORTANT RULES:
1. If information is not in the context, say "I'm sorry, I don't have information about that in my knowledge base."
2. Do not make up or add information not present in the context
3. Use clear and easy-to-understand English
4. Focus on information most relevant to the question
5. Don't start your answer with an opening greeting"""
    
    def _build_user_prompt(self, question: str, context: str, language: str) -> str:
        """Build user prompt with context and question"""
        if language == 'id':
            return f"""Konteks informasi:
{context}

Pertanyaan pengunjung: {question}

Jawaban (dalam 2-4 kalimat):"""
        else:
            return f"""Context information:
{context}

Visitor question: {question}

Answer (in 2-4 sentences):"""
    
    def _postprocess_answer(self, answer: str, language: str) -> str:
        """Post-process generated answer"""
        # Remove common LLM artifacts
        answer = answer.strip()
        
        # Remove potential prompt repetition
        if language == 'id':
            prefixes_to_remove = ['Jawaban:', 'Jawab:', 'Berdasarkan konteks,']
        else:
            prefixes_to_remove = ['Answer:', 'Based on the context,']
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure ends with punctuation
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def test_generation(self, test_prompt: str = None) -> Dict:
        """
        Test LLM generation with a simple prompt
        Useful for debugging and validation
        """
        if test_prompt is None:
            test_prompt = "Jelaskan dalam 2 kalimat apa itu Reog Ponorogo."
        
        logger.info(f"🧪 Testing LLM with: {test_prompt}")
        
        result = self.generate(test_prompt, temperature=0.7, max_tokens=100)
        
        return {
            'prompt': test_prompt,
            'response': result,
            'success': len(result) > 0
        }


# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """Get singleton LLM service instance"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance

# For backward compatibility
llm_service = get_llm_service()