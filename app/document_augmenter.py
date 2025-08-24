"""
SAIA-RAG Document Augmentation Module

Generates hypothetical customer questions for knowledge base articles to improve
retrieval matching and coverage. Implements best practices for query expansion.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

import structlog
from pydantic import BaseModel, Field

from .config import get_settings
from .vector_store import get_vector_store
from .openai_client import get_openai_client

logger = structlog.get_logger()


class HypotheticalQuestion(BaseModel):
    """Generated hypothetical question."""
    question: str = Field(..., description="Generated question text")
    question_type: str = Field(..., description="Type of question (direct, variation, frustrated, technical, simple)")
    confidence: float = Field(..., description="Generation confidence (0.0-1.0)")
    language: str = Field(..., description="Question language (ar, en)")


class DocumentAugmentation(BaseModel):
    """Document augmentation result."""
    document_id: str = Field(..., description="Source document ID")
    chunk_id: str = Field(..., description="Source chunk ID")
    original_text: str = Field(..., description="Original chunk text")
    category: str = Field(..., description="Document category")
    generated_questions: List[HypotheticalQuestion] = Field(..., description="Generated questions")
    augmentation_metadata: Dict[str, Any] = Field(..., description="Augmentation metadata")


class DocumentAugmenter:
    """
    Document augmentation service for improving retrieval coverage.
    
    Generates hypothetical questions that customers might ask for each
    knowledge base article, improving the chances of successful retrieval.
    """
    
    def __init__(self):
        """Initialize document augmenter."""
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.openai_client = get_openai_client()
        
        # Question generation templates by category
        self.question_templates = {
            "FAQ": {
                "direct": "What is {topic}?",
                "how_to": "How do I {action}?",
                "can_i": "Can I {action}?",
                "why": "Why {situation}?",
                "when": "When should I {action}?"
            },
            "troubleshooting": {
                "problem": "I have a problem with {topic}",
                "not_working": "{feature} is not working",
                "error": "I'm getting an error with {topic}",
                "fix": "How do I fix {issue}?",
                "help": "I need help with {topic}"
            },
            "billing": {
                "payment": "How do I pay for {service}?",
                "cost": "How much does {service} cost?",
                "refund": "Can I get a refund for {service}?",
                "billing": "I have a billing question about {topic}",
                "charge": "Why was I charged for {service}?"
            },
            "services": {
                "what_is": "What is {service}?",
                "how_does": "How does {service} work?",
                "available": "Is {service} available?",
                "features": "What features does {service} have?",
                "comparison": "What's the difference between {service1} and {service2}?"
            },
            "policies": {
                "policy": "What is your policy on {topic}?",
                "allowed": "Am I allowed to {action}?",
                "privacy": "How do you handle {data_type}?",
                "rights": "What are my rights regarding {topic}?",
                "terms": "What are the terms for {service}?"
            },
            "support": {
                "contact": "How do I contact support?",
                "help": "I need help with {topic}",
                "assistance": "Can you help me with {issue}?",
                "guidance": "Can you guide me through {process}?",
                "support": "Do you support {feature}?"
            }
        }
        
        logger.info("Document augmenter initialized")
    
    async def augment_all_documents(self) -> List[DocumentAugmentation]:
        """
        Augment all documents in the knowledge base with hypothetical questions.
        
        Returns:
            List of augmentation results
        """
        try:
            logger.info("Starting document augmentation for all documents")
            
            # Get all points from vector store
            all_points = await self._get_all_points()
            
            # Group points by document
            documents = self._group_points_by_document(all_points)
            
            # Augment each document
            augmentation_results = []
            
            for doc_id, chunks in documents.items():
                logger.info(f"Augmenting document: {doc_id} ({len(chunks)} chunks)")
                
                for chunk in chunks:
                    try:
                        augmentation = await self.augment_document_chunk(chunk)
                        augmentation_results.append(augmentation)
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Failed to augment chunk {chunk.id}", error=str(e))
                        continue
            
            logger.info(
                "Document augmentation completed",
                total_chunks=len(all_points),
                augmented_chunks=len(augmentation_results)
            )
            
            return augmentation_results
            
        except Exception as e:
            logger.error("Document augmentation failed", error=str(e))
            raise
    
    async def augment_document_chunk(self, point: Any) -> DocumentAugmentation:
        """
        Augment a single document chunk with hypothetical questions.
        
        Args:
            point: Vector store point with payload
            
        Returns:
            Document augmentation result
        """
        try:
            payload = point.payload
            text = payload.get('text', '')
            category = payload.get('category', 'general')
            title = payload.get('title', 'Unknown')
            
            # Generate questions using LLM
            generated_questions = await self._generate_questions_with_llm(
                text=text,
                category=category,
                title=title
            )
            
            # Create augmentation result
            augmentation = DocumentAugmentation(
                document_id=payload.get('document_id', 'unknown'),
                chunk_id=payload.get('chunk_id', point.id),
                original_text=text,
                category=category,
                generated_questions=generated_questions,
                augmentation_metadata={
                    "augmented_at": datetime.utcnow().isoformat(),
                    "title": title,
                    "chunk_length": len(text),
                    "questions_generated": len(generated_questions)
                }
            )
            
            logger.info(
                "Chunk augmented successfully",
                chunk_id=point.id,
                questions_generated=len(generated_questions),
                category=category
            )
            
            return augmentation
            
        except Exception as e:
            logger.error(f"Failed to augment chunk {point.id}", error=str(e))
            raise
    
    async def _generate_questions_with_llm(
        self,
        text: str,
        category: str,
        title: str
    ) -> List[HypotheticalQuestion]:
        """Generate hypothetical questions using LLM."""
        
        # Detect language
        language = self._detect_language(text)
        
        # Create category-specific prompt
        prompt = self._create_question_generation_prompt(text, category, title, language)
        
        try:
            # Generate questions using OpenAI
            messages = [{"role": "user", "content": prompt}]
            
            result = await self.openai_client.chat_completion(
                messages=messages,
                temperature=0.7,  # Some creativity for question variation
                max_tokens=500
            )
            
            # Parse generated questions
            questions_text = result["content"]
            questions = self._parse_generated_questions(questions_text, language)
            
            return questions
            
        except Exception as e:
            logger.error("LLM question generation failed", error=str(e))
            # Fallback to template-based generation
            return self._generate_questions_with_templates(text, category, language)
    
    def _create_question_generation_prompt(
        self,
        text: str,
        category: str,
        title: str,
        language: str
    ) -> str:
        """Create prompt for question generation."""
        
        if language == "ar":
            prompt = f"""
            بناءً على النص التالي من قسم {category} بعنوان "{title}"، قم بإنشاء 5 أسئلة مختلفة قد يسألها العملاء:

            النص:
            {text}

            اكتب 5 أسئلة متنوعة:
            1. سؤال مباشر
            2. سؤال بصيغة مختلفة
            3. سؤال من عميل محبط
            4. سؤال تقني
            5. سؤال بسيط

            الأسئلة:
            """
        else:
            prompt = f"""
            Based on this {category} content titled "{title}", generate 5 different questions customers might ask:

            Content:
            {text}

            Generate 5 varied questions:
            1. Direct question
            2. Question with different wording
            3. Question from frustrated customer perspective
            4. Technical question
            5. Simple/beginner question

            Questions:
            """
        
        return prompt
    
    def _parse_generated_questions(self, questions_text: str, language: str) -> List[HypotheticalQuestion]:
        """Parse generated questions from LLM response."""
        questions = []
        
        # Split by numbered lines
        lines = questions_text.strip().split('\n')
        question_types = ["direct", "variation", "frustrated", "technical", "simple"]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering (1., 2., etc.)
            question = re.sub(r'^\d+\.?\s*', '', line).strip()
            
            if question and len(question) > 5:  # Valid question
                question_type = question_types[i] if i < len(question_types) else "general"
                
                questions.append(HypotheticalQuestion(
                    question=question,
                    question_type=question_type,
                    confidence=0.8,  # High confidence for LLM-generated
                    language=language
                ))
        
        return questions
    
    def _generate_questions_with_templates(
        self,
        text: str,
        category: str,
        language: str
    ) -> List[HypotheticalQuestion]:
        """Fallback template-based question generation."""
        questions = []
        
        # Extract key terms from text
        key_terms = self._extract_key_terms(text)
        
        # Get templates for category
        templates = self.question_templates.get(category, self.question_templates["support"])
        
        # Generate questions from templates
        for question_type, template in templates.items():
            if key_terms:
                # Use first key term for template
                topic = key_terms[0]
                
                if language == "ar":
                    question = self._translate_template_to_arabic(template, topic)
                else:
                    question = template.format(topic=topic, action=topic, service=topic)
                
                questions.append(HypotheticalQuestion(
                    question=question,
                    question_type=question_type,
                    confidence=0.6,  # Lower confidence for template-based
                    language=language
                ))
        
        return questions[:5]  # Limit to 5 questions
    
    def _detect_language(self, text: str) -> str:
        """Detect text language (Arabic or English)."""
        # Simple Arabic detection
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', text))
        
        if total_chars == 0:
            return "en"  # Default to English
        
        arabic_ratio = arabic_chars / total_chars
        return "ar" if arabic_ratio > 0.5 else "en"
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        key_terms = [word for word in words if word not in stop_words and len(word) > 3]
        
        return key_terms[:10]  # Return top 10 terms
    
    def _translate_template_to_arabic(self, template: str, topic: str) -> str:
        """Simple template translation to Arabic."""
        # Basic template translations
        translations = {
            "What is {topic}?": f"ما هو {topic}؟",
            "How do I {action}?": f"كيف أقوم بـ {topic}؟",
            "Can I {action}?": f"هل يمكنني {topic}؟",
            "I have a problem with {topic}": f"لدي مشكلة مع {topic}",
            "How do I contact support?": "كيف أتواصل مع الدعم؟",
            "I need help with {topic}": f"أحتاج مساعدة مع {topic}"
        }
        
        return translations.get(template, f"سؤال حول {topic}")
    
    async def _get_all_points(self) -> List[Any]:
        """Get all points from the vector store."""
        all_points = []
        offset = None
        
        while True:
            points, next_offset = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            all_points.extend(points)
            
            if next_offset is None:
                break
            offset = next_offset
        
        return all_points
    
    def _group_points_by_document(self, points: List[Any]) -> Dict[str, List[Any]]:
        """Group points by document ID."""
        documents = {}
        
        for point in points:
            doc_id = point.payload.get('document_id', 'unknown')
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(point)
        
        return documents


    async def save_augmentation_results(
        self,
        augmentation_results: List[DocumentAugmentation],
        output_file: str = "docs/document_augmentation_results.json"
    ) -> None:
        """Save augmentation results to file for analysis."""
        import json

        # Convert to serializable format
        results_data = []
        for result in augmentation_results:
            result_dict = {
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "category": result.category,
                "original_text_preview": result.original_text[:200] + "..." if len(result.original_text) > 200 else result.original_text,
                "questions": [
                    {
                        "question": q.question,
                        "type": q.question_type,
                        "confidence": q.confidence,
                        "language": q.language
                    }
                    for q in result.generated_questions
                ],
                "metadata": result.augmentation_metadata
            }
            results_data.append(result_dict)

        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Augmentation results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save augmentation results", error=str(e))

    async def process_knowledge_base_augmentation(self) -> Dict[str, Any]:
        """
        Complete knowledge base augmentation process.

        Returns:
            Summary of augmentation process
        """
        try:
            logger.info("Starting complete knowledge base augmentation")
            start_time = datetime.utcnow()

            # Augment all documents
            augmentation_results = await self.augment_all_documents()

            # Save results
            await self.save_augmentation_results(augmentation_results)

            # Generate summary statistics
            total_questions = sum(len(result.generated_questions) for result in augmentation_results)
            categories = {}
            languages = {}

            for result in augmentation_results:
                # Count by category
                category = result.category
                categories[category] = categories.get(category, 0) + len(result.generated_questions)

                # Count by language
                for question in result.generated_questions:
                    lang = question.language
                    languages[lang] = languages.get(lang, 0) + 1

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            summary = {
                "status": "completed",
                "processing_time_seconds": processing_time,
                "chunks_processed": len(augmentation_results),
                "total_questions_generated": total_questions,
                "average_questions_per_chunk": total_questions / len(augmentation_results) if augmentation_results else 0,
                "questions_by_category": categories,
                "questions_by_language": languages,
                "completed_at": end_time.isoformat()
            }

            logger.info(
                "Knowledge base augmentation completed",
                **summary
            )

            return summary

        except Exception as e:
            logger.error("Knowledge base augmentation failed", error=str(e))
            raise


# Global instance
_document_augmenter = None


def get_document_augmenter() -> DocumentAugmenter:
    """Get global document augmenter instance."""
    global _document_augmenter
    if _document_augmenter is None:
        _document_augmenter = DocumentAugmenter()
    return _document_augmenter
