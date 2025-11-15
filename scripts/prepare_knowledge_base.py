"""
Knowledge Base Preparation Script
Converts raw text documents into structured JSON for embedding
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from loguru import logger


class KnowledgeBasePreprocessor:
    """
    Preprocessor for Reog Ponorogo knowledge base
    
    Features:
    - Text cleaning and normalization
    - Smart text chunking (splits long docs)
    - Keyword extraction
    - Bilingual support (ID + EN)
    - Metadata enrichment
    """
    
    def __init__(self, raw_data_dir: str, output_file: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.documents = []
        
        # Category definitions
        self.categories = {
            'sejarah': 'History',
            'filosofi': 'Philosophy',
            'tokoh': 'Characters',
            'kostum': 'Costumes',
            'tarian': 'Dance',
            'festival': 'Festivals',
            'umkm': 'Local Products',
            'faq': 'FAQ'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        - Remove extra whitespace
        - Normalize punctuation
        - Remove special characters (keep diacritics)
        """
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special chars but keep Indonesian diacritics and punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\—\:\;\'\"]', '', text, flags=re.UNICODE)
        
        # --- FIX: Normalize quotes (smart quotes -> standard) ---
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, max_chunk_size: int = 600, overlap: int = 50, min_chunk_size: int = 100) -> List[str]:
        """
        Split long text into overlapping chunks
        
        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            min_chunk_size: Minimum characters for a chunk (to avoid tiny remnants)
        
        Returns:
            List of text chunks
        """
        # Split into sentences. (Regex: lookbehind for .!? followed by whitespace)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence_to_add = sentence + ' ' # Re-add space split on
            
            # Handle exceptionally long sentences (longer than max_chunk_size)
            if len(sentence_to_add) > max_chunk_size:
                # If there's a chunk being built, save it first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Add the long sentence as its own chunk
                # (This could be split further, but for now, just add it)
                chunks.append(sentence_to_add.strip())
                current_chunk = "" # Reset
                continue # Skip to next sentence

            # Standard case: check if adding the sentence exceeds the max
            if len(current_chunk) + len(sentence_to_add) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                # Take the last ~10 words for overlap (overlap=50 // 5)
                overlap_words = words[-overlap//5:] if len(words) > overlap//5 else words
                # Start new chunk with overlap AND the new sentence
                current_chunk = ' '.join(overlap_words) + ' ' + sentence_to_add
            else:
                # Add sentence to current chunk
                current_chunk += sentence_to_add
        
        # --- LOGIC FIX: Handle the last chunk ---
        
        # Add the remaining chunk, check its length
        final_chunk = current_chunk.strip()
        if final_chunk:
            # If the last chunk is too short AND there are preceding chunks
            if len(final_chunk) < min_chunk_size and len(chunks) > 0:
                # Merge this short chunk into the previous chunk
                # This might make the last chunk > max_chunk_size, 
                # but it's better than a useless tiny chunk.
                logger.debug(f"Merging short final chunk (len {len(final_chunk)}) to previous chunk.")
                chunks[-1] = chunks[-1] + ' ' + final_chunk
            else:
                # The last chunk is long enough, or this is the *only* chunk
                chunks.append(final_chunk)
        
        # --- END OF LOGIC FIX ---
        
        # If no chunking was needed (original text was short)
        if not chunks:
            chunks = [text]
        
        # Final filter to ensure no empty strings
        return [c for c in chunks if c]
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords using frequency analysis
        (Simple TF approach - for production, use TF-IDF or KeyBERT)
        """
        # Stopwords (simplified, includes ID + EN)
        stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan',
            'adalah', 'ini', 'itu', 'atau', 'oleh', 'dalam', 'akan', 'telah',
            'dapat', 'ada', 'sebagai', 'juga', 'tidak', 'mereka', 'kami',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Extract words (lowercase, min length 3)
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top N most frequent
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_n]]
        
        return keywords
    
    def process_txt_file(self, file_path: Path, category: str, language: str):
        """Process a single text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {file_path}")
                return
            
            # Clean content
            content = self.clean_text(content)
            
            # If content after cleaning is still < 100, that's an issue
            # But we assume original files > 100, so after cleaning > 100
            
            # Chunk if necessary
            # The new chunk_text function will use min_chunk_size=100 by default
            chunks = self.chunk_text(content, max_chunk_size=600, overlap=50)
            
            # Extract title from filename
            title = file_path.stem
            # Remove numbering prefix (e.g., "01_" from "01_asal_usul_reog")
            title = re.sub(r'^\d+_', '', title)
            title = title.replace('_', ' ').title()
            
            # Create document entry for each chunk
            for i, chunk in enumerate(chunks):
                
                # --- Additional validation here ---
                if len(chunk) < 100:
                    # This shouldn't happen anymore, but as a safeguard
                    logger.warning(f"Generated chunk is too short ({len(chunk)} chars) from {file_path.name} - Chunk {i}")
                    # Skip this chunk so it doesn't enter the knowledge base
                    continue 

                doc_id = f"doc_{len(self.documents) + 1:04d}"
                
                # Extract keywords
                keywords = self.extract_keywords(chunk)
                
                document = {
                    "id": doc_id,
                    "category": category,
                    "title": f"{title} - Part {i+1}" if len(chunks) > 1 else title,
                    "language": language,
                    "content": chunk,
                    "keywords": keywords,
                    "metadata": {
                        "source_file": file_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "word_count": len(chunk.split()),
                        "char_count": len(chunk)
                    }
                }
                
                self.documents.append(document)
                
            logger.info(f"✅ Processed: {file_path.name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
    
    def process_directory(self):
        """Process all files in raw_data_dir"""
        logger.info("="*70)
        logger.info("KNOWLEDGE BASE PREPROCESSING")
        logger.info("="*70)
        
        if not self.raw_data_dir.exists():
            logger.error(f"Raw data directory not found: {self.raw_data_dir}")
            logger.info(f"Please create: {self.raw_data_dir}")
            return
        
        for category_dir, category_name in self.categories.items():
            category_path = self.raw_data_dir / category_dir
            
            if not category_path.exists():
                logger.warning(f"Category directory not found: {category_path}")
                continue
            
            logger.info(f"\n📁 Processing category: {category_name}")
            logger.info("-"*70)
            
            # Process Indonesian files
            id_dir = category_path / 'id'
            if id_dir.exists():
                txt_files = sorted(id_dir.glob('*.txt'))
                logger.info(f"Indonesian files: {len(txt_files)}")
                for txt_file in txt_files:
                    self.process_txt_file(txt_file, category_dir, 'id')
            else:
                logger.warning(f"Indonesian directory not found: {id_dir}")
            
            # Process English files
            en_dir = category_path / 'en'
            if en_dir.exists():
                txt_files = sorted(en_dir.glob('*.txt'))
                logger.info(f"English files: {len(txt_files)}")
                for txt_file in txt_files:
                    self.process_txt_file(txt_file, category_dir, 'en')
            else:
                logger.warning(f"English directory not found: {en_dir}")
    
    def save_knowledge_base(self):
        """Save processed documents to JSON"""
        if not self.documents:
            logger.error("❌ No documents to save!")
            return
        
        # Calculate statistics
        categories_count = {}
        languages_count = {}
        
        for doc in self.documents:
            cat = doc['category']
            lang = doc['language']
            categories_count[cat] = categories_count.get(cat, 0) + 1
            languages_count[lang] = languages_count.get(lang, 0) + 1
        
        # Prepare output data
        output_data = {
            "metadata": {
                "total_documents": len(self.documents),
                "categories": list(categories_count.keys()),
                "languages": list(languages_count.keys()),
                "statistics": {
                    "by_category": categories_count,
                    "by_language": languages_count
                }
            },
            "documents": self.documents
        }
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n✅ Knowledge base saved to: {self.output_file}")
        logger.info(f"📊 Total documents: {len(self.documents)}")
        logger.info(f"📂 Categories: {len(categories_count)}")
        logger.info(f"🌐 Languages: {len(languages_count)}")
        logger.info("\nDocument distribution:")
        for cat, count in categories_count.items():
            logger.info(f"   {cat}: {count} docs")
        logger.info("\nLanguage distribution:")
        for lang, count in languages_count.items():
            logger.info(f"   {lang}: {count} docs")
    
    def validate_knowledge_base(self):
        """Validate the generated knowledge base"""
        logger.info("\n🔍 VALIDATING KNOWLEDGE BASE...")
        logger.info("-"*70)
        
        issues = []
        
        # Check minimum documents
        if len(self.documents) < 30:
            issues.append(f"⚠️ Only {len(self.documents)} documents (recommended: 50+)")
        
        # Check bilingual coverage
        id_docs = sum(1 for d in self.documents if d['language'] == 'id')
        en_docs = sum(1 for d in self.documents if d['language'] == 'en')
        
        if id_docs < 20:
            issues.append(f"⚠️ Only {id_docs} Indonesian documents")
        if en_docs < 20:
            issues.append(f"⚠️ Only {en_docs} English documents")
        
        # Check content quality
        short_docs = [d for d in self.documents if len(d['content']) < 100]
        if short_docs:
            issues.append(f"⚠️ {len(short_docs)} documents too short (< 100 chars)")
            # Show details for debugging
            for d in short_docs[:5]: # Show the first 5 problematic ones
                logger.warning(f"   - Short doc: {d['id']} (len: {len(d['content'])}) from {d['metadata']['source_file']}")
        
        # Report
        if issues:
            logger.warning("Issues found:")
            for issue in issues:
                logger.warning(f"  {issue}")
        else:
            logger.info("✅ Knowledge base looks good!")
        
        return len(issues) == 0


# Main execution
if __name__ == "__main__":
    processor = KnowledgeBasePreprocessor(
        raw_data_dir="data/raw",
        output_file="data/processed/knowledge_base.json"
    )
    
    processor.process_directory()
    processor.save_knowledge_base()
    processor.validate_knowledge_base()