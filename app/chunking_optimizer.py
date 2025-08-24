"""
SAIA-RAG Chunking Optimization Module

Implements customer support optimized text splitting with proper chunk sizes,
overlap, and context preservation for better retrieval accuracy.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .config import get_settings

logger = structlog.get_logger()


class ChunkType(str, Enum):
    """Types of content chunks."""
    FAQ_ITEM = "faq_item"
    POLICY_SECTION = "policy_section"
    PROCEDURE = "procedure"
    DEFINITION = "definition"
    GENERAL = "general"


class OptimizedChunk(BaseModel):
    """Optimized chunk with enhanced metadata."""
    text: str = Field(..., description="Chunk text content")
    chunk_type: ChunkType = Field(..., description="Type of chunk")
    char_count: int = Field(..., description="Character count")
    sentence_count: int = Field(..., description="Number of sentences")
    context_preserved: bool = Field(..., description="Whether context is preserved")
    boundary_quality: float = Field(..., description="Quality of chunk boundaries (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChunkingOptimizer:
    """
    Advanced chunking optimizer for customer support content.
    
    Implements:
    - Category-specific chunking strategies
    - Semantic boundary detection
    - Context preservation
    - Quality scoring
    """
    
    def __init__(self):
        """Initialize chunking optimizer."""
        self.settings = get_settings()
        
        # Category-specific chunk sizes (optimized for customer support)
        self.category_configs = {
            "FAQ": {
                "target_size": 600,  # Smaller for focused FAQ items
                "max_size": 900,
                "min_size": 200,
                "overlap": 100,
                "preserve_qa_pairs": True
            },
            "troubleshooting": {
                "target_size": 800,  # Medium for step-by-step guides
                "max_size": 1200,
                "min_size": 300,
                "overlap": 150,
                "preserve_steps": True
            },
            "policies": {
                "target_size": 700,  # Medium for policy sections
                "max_size": 1000,
                "min_size": 250,
                "overlap": 150,
                "preserve_sections": True
            },
            "services": {
                "target_size": 650,  # Medium for service descriptions
                "max_size": 950,
                "min_size": 250,
                "overlap": 130,
                "preserve_features": True
            },
            "billing": {
                "target_size": 550,  # Smaller for precise billing info
                "max_size": 800,
                "min_size": 200,
                "overlap": 100,
                "preserve_procedures": True
            },
            "support": {
                "target_size": 600,  # Medium for general support
                "max_size": 900,
                "min_size": 250,
                "overlap": 120,
                "preserve_context": True
            },
            "terms and conditions": {
                "target_size": 750,  # Larger for legal text
                "max_size": 1100,
                "min_size": 300,
                "overlap": 150,
                "preserve_clauses": True
            }
        }
        
        # Default configuration
        self.default_config = {
            "target_size": 650,
            "max_size": 950,
            "min_size": 250,
            "overlap": 130,
            "preserve_context": True
        }
        
        # Arabic text patterns for better boundary detection
        self.arabic_sentence_endings = [
            '.',  # Period
            '؟',  # Arabic question mark
            '!',  # Exclamation mark
            '؛',  # Arabic semicolon
            ':',  # Colon
            '،',  # Arabic comma (sometimes used as sentence ending)
        ]
        
        # FAQ patterns for Q&A pair detection
        self.qa_patterns = [
            r'سؤال\s*\d*\s*[:：]',  # Question patterns
            r'س\s*\d*\s*[:：]',     # Short question
            r'جواب\s*\d*\s*[:：]',  # Answer patterns
            r'ج\s*\d*\s*[:：]',     # Short answer
            r'\d+\)\s*',           # Numbered items
            r'faq\d*[QA]\s*[:：]', # English FAQ patterns
        ]
        
        logger.info("Chunking optimizer initialized")
    
    def optimize_chunks(
        self,
        text: str,
        category: str = "general",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[OptimizedChunk]:
        """
        Create optimized chunks for customer support content.
        
        Args:
            text: Text to chunk
            category: Content category
            title: Document title
            metadata: Additional metadata
            
        Returns:
            List of optimized chunks
        """
        try:
            # Get category-specific configuration
            config = self._get_category_config(category)
            
            # Detect content structure
            content_structure = self._analyze_content_structure(text, category)
            
            # Apply category-specific chunking strategy
            if category.lower() == "faq":
                chunks = self._chunk_faq_content(text, config, content_structure)
            elif category.lower() in ["policies", "سياسة الخصوصية"]:
                chunks = self._chunk_policy_content(text, config, content_structure)
            elif category.lower() in ["terms and conditions", "شروط و احكام"]:
                chunks = self._chunk_legal_content(text, config, content_structure)
            else:
                chunks = self._chunk_general_content(text, config, content_structure)
            
            # Enhance chunks with metadata
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = self._enhance_chunk_metadata(
                    chunk, i, category, title, metadata
                )
                enhanced_chunks.append(enhanced_chunk)
            
            # Validate and optimize chunk boundaries
            optimized_chunks = self._optimize_chunk_boundaries(enhanced_chunks, config)
            
            logger.info(
                "Chunks optimized successfully",
                original_length=len(text),
                chunks_created=len(optimized_chunks),
                category=category,
                avg_chunk_size=sum(c.char_count for c in optimized_chunks) // len(optimized_chunks) if optimized_chunks else 0
            )
            
            return optimized_chunks
            
        except Exception as e:
            logger.error("Chunk optimization failed", error=str(e))
            # Fallback to basic chunking
            return self._fallback_chunking(text, category, metadata)
    
    def _get_category_config(self, category: str) -> Dict[str, Any]:
        """Get configuration for specific category."""
        # Normalize category name
        category_lower = category.lower()
        
        # Map Arabic categories to English
        category_mapping = {
            "سياسة الخصوصية": "policies",
            "شروط و احكام": "terms and conditions",
            "خدمات": "services",
            "عن الشركة": "support"
        }
        
        mapped_category = category_mapping.get(category_lower, category_lower)
        return self.category_configs.get(mapped_category, self.default_config)
    
    def _analyze_content_structure(self, text: str, category: str) -> Dict[str, Any]:
        """Analyze content structure for better chunking."""
        structure = {
            "has_qa_pairs": False,
            "has_numbered_items": False,
            "has_sections": False,
            "has_procedures": False,
            "sentence_count": 0,
            "paragraph_count": 0,
            "language": "ar" if self._is_arabic_text(text) else "en"
        }
        
        # Count sentences and paragraphs
        sentences = self._split_into_sentences(text)
        structure["sentence_count"] = len(sentences)
        structure["paragraph_count"] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Detect Q&A pairs
        for pattern in self.qa_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure["has_qa_pairs"] = True
                break
        
        # Detect numbered items
        if re.search(r'\d+[.)]\s+', text):
            structure["has_numbered_items"] = True
        
        # Detect sections (headers, titles)
        if re.search(r'^[A-Za-z\u0600-\u06FF\s]+:$', text, re.MULTILINE):
            structure["has_sections"] = True
        
        # Detect procedures (step-by-step)
        procedure_indicators = ['خطوة', 'step', 'أولاً', 'ثانياً', 'first', 'then', 'next']
        if any(indicator in text.lower() for indicator in procedure_indicators):
            structure["has_procedures"] = True
        
        return structure
    
    def _chunk_faq_content(
        self,
        text: str,
        config: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk FAQ content preserving Q&A pairs."""
        chunks = []
        
        if structure["has_qa_pairs"]:
            # Split by Q&A patterns
            qa_splits = self._split_by_qa_patterns(text)
            
            current_chunk = ""
            for qa_item in qa_splits:
                # If adding this Q&A would exceed max size, finalize current chunk
                if len(current_chunk + qa_item) > config["max_size"] and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_type": ChunkType.FAQ_ITEM,
                        "boundary_quality": 0.9  # High quality for Q&A boundaries
                    })
                    current_chunk = qa_item
                else:
                    current_chunk += "\n\n" + qa_item if current_chunk else qa_item
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_type": ChunkType.FAQ_ITEM,
                    "boundary_quality": 0.9
                })
        else:
            # Fallback to sentence-based chunking
            chunks = self._chunk_by_sentences(text, config, ChunkType.FAQ_ITEM)
        
        return chunks
    
    def _chunk_policy_content(
        self,
        text: str,
        config: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk policy content preserving sections."""
        if structure["has_sections"]:
            # Split by sections
            sections = re.split(r'\n\s*([A-Za-z\u0600-\u06FF\s]+:)\s*\n', text)
            chunks = []
            
            current_chunk = ""
            for section in sections:
                if len(current_chunk + section) > config["max_size"] and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_type": ChunkType.POLICY_SECTION,
                        "boundary_quality": 0.8
                    })
                    current_chunk = section
                else:
                    current_chunk += "\n" + section if current_chunk else section
            
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_type": ChunkType.POLICY_SECTION,
                    "boundary_quality": 0.8
                })
            
            return chunks
        else:
            return self._chunk_by_sentences(text, config, ChunkType.POLICY_SECTION)
    
    def _chunk_legal_content(
        self,
        text: str,
        config: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk legal content preserving clauses."""
        # Legal content often has numbered clauses
        if structure["has_numbered_items"]:
            clauses = re.split(r'\n\s*(\d+[.)]\s+)', text)
            chunks = []
            
            current_chunk = ""
            for clause in clauses:
                if len(current_chunk + clause) > config["max_size"] and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_type": ChunkType.DEFINITION,
                        "boundary_quality": 0.85
                    })
                    current_chunk = clause
                else:
                    current_chunk += clause
            
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_type": ChunkType.DEFINITION,
                    "boundary_quality": 0.85
                })
            
            return chunks
        else:
            return self._chunk_by_sentences(text, config, ChunkType.DEFINITION)
    
    def _chunk_general_content(
        self,
        text: str,
        config: Dict[str, Any],
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk general content with optimal boundaries."""
        if structure["has_procedures"]:
            return self._chunk_by_sentences(text, config, ChunkType.PROCEDURE)
        else:
            return self._chunk_by_sentences(text, config, ChunkType.GENERAL)
    
    def _chunk_by_sentences(
        self,
        text: str,
        config: Dict[str, Any],
        chunk_type: ChunkType
    ) -> List[Dict[str, Any]]:
        """Chunk text by sentences with optimal boundaries."""
        sentences = self._split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed target size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) > config["target_size"] and current_chunk:
                # Finalize current chunk
                boundary_quality = self._calculate_boundary_quality(current_sentences)
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_type": chunk_type,
                    "boundary_quality": boundary_quality
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                # Add sentence to current chunk
                current_chunk = test_chunk
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            boundary_quality = self._calculate_boundary_quality(current_sentences)
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_type": chunk_type,
                "boundary_quality": boundary_quality
            })
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with Arabic support."""
        # Create pattern for sentence endings
        endings_pattern = '|'.join(re.escape(ending) for ending in self.arabic_sentence_endings)

        # Split by sentence endings
        sentences = re.split(f'({endings_pattern})\\s+', text)

        # Reconstruct sentences with their endings
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                ending = sentences[i + 1]
                sentence += ending

            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                reconstructed.append(sentence)

        return reconstructed

    def _split_by_qa_patterns(self, text: str) -> List[str]:
        """Split text by Q&A patterns."""
        qa_items = []

        # Try to split by Q&A patterns
        for pattern in self.qa_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Split by this pattern
                last_end = 0
                for match in matches:
                    if last_end < match.start():
                        # Add content before this match
                        content = text[last_end:match.start()].strip()
                        if content:
                            qa_items.append(content)

                    # Find the end of this Q&A item (next pattern or end of text)
                    next_match = None
                    for next_pattern in self.qa_patterns:
                        next_matches = list(re.finditer(next_pattern, text[match.end():], re.IGNORECASE))
                        if next_matches:
                            if next_match is None or next_matches[0].start() < next_match.start():
                                next_match = next_matches[0]

                    if next_match:
                        qa_content = text[match.start():match.end() + next_match.start()].strip()
                    else:
                        qa_content = text[match.start():].strip()

                    if qa_content:
                        qa_items.append(qa_content)

                    last_end = match.end() + (next_match.start() if next_match else len(text) - match.end())

                break  # Use first matching pattern

        # If no patterns found, split by paragraphs
        if not qa_items:
            qa_items = [p.strip() for p in text.split('\n\n') if p.strip()]

        return qa_items

    def _calculate_boundary_quality(self, sentences: List[str]) -> float:
        """Calculate quality of chunk boundaries."""
        if not sentences:
            return 0.0

        quality_score = 0.5  # Base score

        # Check if chunk ends with complete sentence
        last_sentence = sentences[-1].strip()
        if any(last_sentence.endswith(ending) for ending in self.arabic_sentence_endings):
            quality_score += 0.2

        # Check if chunk has good sentence count (not too few, not too many)
        sentence_count = len(sentences)
        if 3 <= sentence_count <= 8:
            quality_score += 0.2
        elif sentence_count < 2:
            quality_score -= 0.2

        # Check for coherence indicators
        coherence_indicators = ['لذلك', 'وبالتالي', 'كما', 'أيضاً', 'بالإضافة', 'however', 'therefore', 'also']
        if any(indicator in ' '.join(sentences).lower() for indicator in coherence_indicators):
            quality_score += 0.1

        return min(1.0, max(0.0, quality_score))

    def _enhance_chunk_metadata(
        self,
        chunk: Dict[str, Any],
        index: int,
        category: str,
        title: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> OptimizedChunk:
        """Enhance chunk with comprehensive metadata."""
        text = chunk["text"]

        # Count sentences
        sentences = self._split_into_sentences(text)
        sentence_count = len(sentences)

        # Determine context preservation
        context_preserved = chunk.get("boundary_quality", 0.5) > 0.7

        # Create enhanced metadata
        enhanced_metadata = {
            "chunk_index": index,
            "category": category,
            "title": title,
            "language": "ar" if self._is_arabic_text(text) else "en",
            "has_questions": any(ending in text for ending in ['؟', '?']),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_procedures": any(word in text.lower() for word in ['خطوة', 'step', 'أولاً', 'ثانياً']),
            **(metadata or {})
        }

        return OptimizedChunk(
            text=text,
            chunk_type=chunk["chunk_type"],
            char_count=len(text),
            sentence_count=sentence_count,
            context_preserved=context_preserved,
            boundary_quality=chunk.get("boundary_quality", 0.5),
            metadata=enhanced_metadata
        )

    def _optimize_chunk_boundaries(
        self,
        chunks: List[OptimizedChunk],
        config: Dict[str, Any]
    ) -> List[OptimizedChunk]:
        """Optimize chunk boundaries for better context preservation."""
        if len(chunks) <= 1:
            return chunks

        optimized = []
        skip_next = False

        for i, chunk in enumerate(chunks):
            if skip_next:
                skip_next = False
                continue

            # Check if chunk is too small and can be merged
            if (chunk.char_count < config["min_size"] and
                i < len(chunks) - 1 and
                chunk.char_count + chunks[i + 1].char_count <= config["max_size"]):

                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_text = chunk.text + "\n\n" + next_chunk.text

                merged_chunk = OptimizedChunk(
                    text=merged_text,
                    chunk_type=chunk.chunk_type,
                    char_count=len(merged_text),
                    sentence_count=chunk.sentence_count + next_chunk.sentence_count,
                    context_preserved=True,  # Merging preserves context
                    boundary_quality=(chunk.boundary_quality + next_chunk.boundary_quality) / 2,
                    metadata={**chunk.metadata, "merged": True}
                )

                optimized.append(merged_chunk)
                skip_next = True  # Skip next chunk as it's been merged
            else:
                optimized.append(chunk)

        return optimized

    def _is_arabic_text(self, text: str) -> bool:
        """Check if text is primarily Arabic."""
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))

        if total_chars == 0:
            return False

        return (arabic_chars / total_chars) > 0.5

    def _fallback_chunking(
        self,
        text: str,
        category: str,
        metadata: Optional[Dict[str, Any]]
    ) -> List[OptimizedChunk]:
        """Fallback chunking method when optimization fails."""
        config = self._get_category_config(category)

        # Simple sentence-based chunking
        sentences = self._split_into_sentences(text)
        chunks = []

        current_chunk = ""
        sentence_count = 0

        for sentence in sentences:
            if len(current_chunk + sentence) > config["target_size"] and current_chunk:
                chunks.append(OptimizedChunk(
                    text=current_chunk.strip(),
                    chunk_type=ChunkType.GENERAL,
                    char_count=len(current_chunk.strip()),
                    sentence_count=sentence_count,
                    context_preserved=False,
                    boundary_quality=0.5,
                    metadata=metadata or {}
                ))
                current_chunk = sentence
                sentence_count = 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                sentence_count += 1

        if current_chunk.strip():
            chunks.append(OptimizedChunk(
                text=current_chunk.strip(),
                chunk_type=ChunkType.GENERAL,
                char_count=len(current_chunk.strip()),
                sentence_count=sentence_count,
                context_preserved=False,
                boundary_quality=0.5,
                metadata=metadata or {}
            ))

        return chunks

    def analyze_chunking_performance(
        self,
        original_chunks: List[Dict[str, Any]],
        optimized_chunks: List[OptimizedChunk]
    ) -> Dict[str, Any]:
        """Analyze performance improvement from optimization."""
        if not original_chunks or not optimized_chunks:
            return {"error": "No chunks to analyze"}

        # Original statistics
        orig_sizes = [len(chunk.get("text", "")) for chunk in original_chunks]
        orig_avg_size = sum(orig_sizes) / len(orig_sizes)
        orig_size_variance = sum((size - orig_avg_size) ** 2 for size in orig_sizes) / len(orig_sizes)

        # Optimized statistics
        opt_sizes = [chunk.char_count for chunk in optimized_chunks]
        opt_avg_size = sum(opt_sizes) / len(opt_sizes)
        opt_size_variance = sum((size - opt_avg_size) ** 2 for size in opt_sizes) / len(opt_sizes)

        # Quality metrics
        avg_boundary_quality = sum(chunk.boundary_quality for chunk in optimized_chunks) / len(optimized_chunks)
        context_preserved_count = sum(1 for chunk in optimized_chunks if chunk.context_preserved)
        context_preservation_rate = context_preserved_count / len(optimized_chunks)

        return {
            "original_chunks": len(original_chunks),
            "optimized_chunks": len(optimized_chunks),
            "size_improvement": {
                "original_avg_size": orig_avg_size,
                "optimized_avg_size": opt_avg_size,
                "size_consistency_improvement": orig_size_variance - opt_size_variance
            },
            "quality_metrics": {
                "avg_boundary_quality": avg_boundary_quality,
                "context_preservation_rate": context_preservation_rate,
                "chunks_with_good_boundaries": sum(1 for c in optimized_chunks if c.boundary_quality > 0.7)
            },
            "chunk_type_distribution": {
                chunk_type.value: sum(1 for c in optimized_chunks if c.chunk_type == chunk_type)
                for chunk_type in ChunkType
            }
        }


# Global instance
_chunking_optimizer = None


def get_chunking_optimizer() -> ChunkingOptimizer:
    """Get global chunking optimizer instance."""
    global _chunking_optimizer
    if _chunking_optimizer is None:
        _chunking_optimizer = ChunkingOptimizer()
    return _chunking_optimizer
