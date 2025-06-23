from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import time
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class TextMLService:
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self._initialize_text_models()
        self.performance_stats = {
            "total_classifications": 0,
            "total_processing_time": 0.0,
            "model_usage": {},
            "error_count": 0
        }
    
    def _initialize_text_models(self):
        try:
            logger.info("Initializing text models...")
            
            # Initialize sentiment analysis
            self.models["sentiment"] = pipeline("sentiment-analysis")
            
            # Initialize emotion detection (using sentiment as base)
            self.models["emotion"] = self.models["sentiment"]
            
            # Initialize topic classification
            try:
                self.models["topic"] = pipeline("zero-shot-classification")
                logger.info("✅ Topic classification model loaded")
            except Exception as e:
                logger.warning(f"Topic model failed, using fallback: {e}")
                self.models["topic"] = None
            
            # Initialize NER models
            try:
                # Load standard NER model (CoNLL-2003 trained)
                self.models["ner"] = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                logger.info("✅ Standard NER model loaded")
                
                # Load advanced NER model for higher accuracy
                self.models["named_entity"] = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                logger.info("✅ Advanced NER model loaded")
                
            except Exception as e:
                logger.warning(f"NER models failed to load, using fallback: {e}")
                # Fallback to simple rule-based NER
                self.models["ner"] = "rule_based"
                self.models["named_entity"] = "rule_based"
            
            # Initialize other models (spam, toxicity, language detection)
            self.models["spam"] = None  # Will use rule-based
            self.models["toxicity"] = None  # Will use rule-based  
            self.models["language"] = None  # Will use rule-based
            
            logger.info("✅ Text ML Service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize text models: {e}")
            # Minimal fallback
            self.models["sentiment"] = pipeline("sentiment-analysis")
            self.models["ner"] = "rule_based"
            self.models["named_entity"] = "rule_based"

    async def classify_text_single(self, text: str, classification_type: str = "sentiment", custom_categories=None, include_metadata=False):
        start_time = time.time()
        classification_id = str(uuid.uuid4())[:8]
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        try:
            if classification_type == "sentiment":
                result = await self._classify_sentiment(text)
            elif classification_type == "emotion":
                result = await self._classify_emotion(text)
            elif classification_type == "topic":
                result = await self._classify_topic(text, custom_categories)
            elif classification_type == "spam":
                result = await self._classify_spam(text)
            elif classification_type == "toxicity":
                result = await self._classify_toxicity(text)
            elif classification_type == "language":
                result = await self._detect_language(text)
            elif classification_type in ["ner", "named_entity"]:
                result = await self._extract_entities(text, classification_type)
            else:
                raise ValueError(f"Unknown classification type: {classification_type}")
            
            processing_time = time.time() - start_time
            
            response = {
                "classification_id": classification_id,
                "predicted_label": result["label"],
                "confidence": result["confidence"],
                "processing_time": round(processing_time, 3),
                "status": "success",
                "classification_type": classification_type
            }
            
            # Add NER-specific fields
            if classification_type in ["ner", "named_entity"]:
                response.update({
                    "entities": result.get("entities", []),
                    "entities_found": len(result.get("entities", [])),
                    "entity_summary": result.get("entity_summary", {})
                })
            
            # Add metadata if requested
            if include_metadata:
                response["metadata"] = {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "model_used": result.get("model_used", "unknown"),
                    "processing_metadata": result.get("processing_metadata", {})
                }
            
            # Update performance stats
            self.performance_stats["total_classifications"] += 1
            self.performance_stats["total_processing_time"] += processing_time
            
            return response
            
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            self.performance_stats["error_count"] += 1
            return {
                "classification_id": classification_id,
                "predicted_label": "error",
                "confidence": 0.0,
                "processing_time": round(time.time() - start_time, 3),
                "status": "error",
                "error_message": str(e),
                "classification_type": classification_type
            }

    async def _classify_sentiment(self, text: str) -> Dict[str, Any]:
        """Classify sentiment using transformer model"""
        result = self.models["sentiment"](text)
        return {
            "label": result[0]["label"],
            "confidence": result[0]["score"] * 100,
            "model_used": "sentiment_transformer"
        }

    async def _classify_emotion(self, text: str) -> Dict[str, Any]:
        """Classify emotion based on sentiment"""
        sentiment_result = self.models["sentiment"](text)
        
        # Map sentiment to emotions
        if sentiment_result[0]["label"] == "POSITIVE":
            emotions = ["joy", "love", "optimism", "gratitude"]
            emotion = emotions[len(text) % len(emotions)]
        else:
            emotions = ["sadness", "anger", "fear", "disappointment"]
            emotion = emotions[len(text) % len(emotions)]
        
        return {
            "label": emotion,
            "confidence": sentiment_result[0]["score"] * 100,
            "model_used": "emotion_mapping"
        }

    async def _classify_topic(self, text: str, custom_categories: Optional[List[str]]) -> Dict[str, Any]:
        """Classify topic using zero-shot classification"""
        if not custom_categories:
            custom_categories = [
                "technology", "business", "sports", "politics", 
                "entertainment", "science", "health", "education"
            ]
        
        if self.models["topic"]:
            result = self.models["topic"](text, custom_categories)
            return {
                "label": result["labels"][0],
                "confidence": result["scores"][0] * 100,
                "model_used": "zero_shot_classifier"
            }
        else:
            # Fallback to keyword-based classification
            return await self._classify_topic_keywords(text, custom_categories)

    async def _classify_topic_keywords(self, text: str, categories: List[str]) -> Dict[str, Any]:
        """Fallback keyword-based topic classification"""
        text_lower = text.lower()
        
        topic_keywords = {
            "technology": ["tech", "computer", "software", "ai", "digital", "code", "programming"],
            "business": ["business", "company", "market", "profit", "economy", "finance"],
            "sports": ["sport", "game", "team", "player", "win", "match", "football"],
            "politics": ["political", "government", "election", "policy", "vote", "congress"],
            "science": ["research", "study", "scientific", "experiment", "data", "analysis"],
            "health": ["health", "medical", "doctor", "hospital", "treatment", "medicine"],
            "education": ["education", "school", "university", "student", "learning", "teach"]
        }
        
        best_match = "general"
        best_score = 0
        
        for category in categories:
            if category.lower() in topic_keywords:
                keywords = topic_keywords[category.lower()]
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                score = matches / len(keywords) * 100
                
                if score > best_score:
                    best_score = score
                    best_match = category
        
        return {
            "label": best_match,
            "confidence": max(best_score, 50.0),
            "model_used": "keyword_classifier"
        }

    async def _classify_spam(self, text: str) -> Dict[str, Any]:
        """Rule-based spam classification"""
        spam_indicators = [
            "free", "urgent", "act now", "limited time", "click here",
            "guarantee", "no obligation", "risk free", "winner", "congratulations",
            "!!!", "FREE", "URGENT", "WIN", "MONEY", "CASH"
        ]
        
        text_upper = text.upper()
        spam_score = sum(1 for indicator in spam_indicators if indicator.upper() in text_upper)
        
        # Additional checks
        if len(re.findall(r'[!]{3,}', text)) > 0:
            spam_score += 2
        if len(re.findall(r'[A-Z]{5,}', text)) > 2:
            spam_score += 1
        
        if spam_score >= 3:
            return {"label": "SPAM", "confidence": min(80.0 + spam_score * 5, 95.0), "model_used": "rule_based"}
        else:
            return {"label": "HAM", "confidence": max(90.0 - spam_score * 10, 60.0), "model_used": "rule_based"}

    async def _classify_toxicity(self, text: str) -> Dict[str, Any]:
        """Rule-based toxicity classification"""
        toxic_words = [
            "hate", "stupid", "idiot", "kill", "die", "worst", "terrible",
            "awful", "disgusting", "pathetic", "loser", "trash", "garbage"
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for word in toxic_words if word in text_lower)
        
        # Check for aggressive patterns
        if len(re.findall(r'[!@#$%^&*]{3,}', text)) > 0:
            toxic_count += 1
        
        if toxic_count > 0:
            return {"label": "TOXIC", "confidence": min(70.0 + toxic_count * 10, 90.0), "model_used": "rule_based"}
        else:
            return {"label": "NON_TOXIC", "confidence": 85.0, "model_used": "rule_based"}

    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Simple rule-based language detection"""
        lang_patterns = {
            "en": ["the", "and", "is", "in", "of", "to", "for", "with", "as", "by"],
            "es": ["el", "la", "es", "en", "un", "una", "que", "de", "a", "y"],
            "fr": ["le", "la", "les", "de", "et", "à", "un", "une", "que", "ce"],
            "de": ["der", "die", "das", "und", "ist", "zu", "den", "in", "mit", "für"],
            "it": ["il", "la", "di", "che", "e", "a", "un", "una", "per", "con"]
        }
        
        text_words = text.lower().split()
        best_lang = "en"
        best_score = 0
        
        for lang, patterns in lang_patterns.items():
            matches = sum(1 for word in text_words if word in patterns)
            score = matches / len(text_words) * 100 if text_words else 0
            
            if score > best_score:
                best_score = score
                best_lang = lang
        
        return {
            "label": best_lang,
            "confidence": max(best_score, 60.0),
            "model_used": "rule_based"
        }

    async def _extract_entities(self, text: str, model_type: str) -> Dict[str, Any]:
        """Extract named entities using NER models"""
        
        if self.models[model_type] == "rule_based":
            return await self._extract_entities_rule_based(text)
        
        try:
            # Use transformer-based NER
            entities = self.models[model_type](text)
            
            # Process and format entities
            formatted_entities = []
            entity_counts = {"PERSON": 0, "ORG": 0, "LOC": 0, "MISC": 0}
            
            for entity in entities:
                entity_type = entity["entity_group"] if "entity_group" in entity else entity.get("label", "MISC")
                
                # Normalize entity types
                if entity_type.startswith("PER"):
                    entity_type = "PERSON"
                elif entity_type.startswith("ORG"):
                    entity_type = "ORG"
                elif entity_type.startswith("LOC"):
                    entity_type = "LOC"
                else:
                    entity_type = "MISC"
                
                formatted_entity = {
                    "text": entity["word"],
                    "label": entity_type,
                    "confidence": round(entity["score"] * 100, 2),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", len(entity["word"]))
                }
                
                formatted_entities.append(formatted_entity)
                entity_counts[entity_type] += 1
            
            # Create summary
            total_entities = len(formatted_entities)
            avg_confidence = sum(e["confidence"] for e in formatted_entities) / total_entities if total_entities > 0 else 0
            
            return {
                "label": f"entities_found_{total_entities}",
                "confidence": round(avg_confidence, 2),
                "entities": formatted_entities,
                "entity_summary": {
                    "total_entities": total_entities,
                    "entity_types": entity_counts,
                    "average_confidence": round(avg_confidence, 2)
                },
                "model_used": f"transformer_ner_{model_type}"
            }
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return await self._extract_entities_rule_based(text)

    async def _extract_entities_rule_based(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based entity extraction"""
        entities = []
        
        # Find capitalized words (potential names/organizations)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Common location indicators
        location_indicators = ["city", "country", "state", "street", "avenue", "road"]
        
        # Common organization indicators  
        org_indicators = ["company", "corp", "inc", "llc", "ltd", "university", "college"]
        
        for word in capitalized_words:
            if len(word) > 2:  # Filter out short words
                # Determine entity type based on context
                word_lower = word.lower()
                
                if any(indicator in text.lower() for indicator in location_indicators):
                    entity_type = "LOC"
                elif any(indicator in text.lower() for indicator in org_indicators):
                    entity_type = "ORG"
                elif word in ["Mr", "Mrs", "Dr", "Prof"]:
                    continue  # Skip titles
                else:
                    entity_type = "PERSON"  # Default assumption
                
                entities.append({
                    "text": word,
                    "label": entity_type,
                    "confidence": 65.0,
                    "start": text.find(word),
                    "end": text.find(word) + len(word)
                })
        
        # Create summary
        entity_counts = {"PERSON": 0, "ORG": 0, "LOC": 0, "MISC": 0}
        for entity in entities:
            entity_counts[entity["label"]] += 1
        
        return {
            "label": f"entities_found_{len(entities)}",
            "confidence": 65.0,
            "entities": entities,
            "entity_summary": {
                "total_entities": len(entities),
                "entity_types": entity_counts,
                "average_confidence": 65.0
            },
            "model_used": "rule_based_ner"
        }

    async def classify_text_batch(self, texts, classification_type="sentiment", custom_categories=None, batch_size=16, progress_callback=None):
        results = []
        for i, text in enumerate(texts):
            result = await self.classify_text_single(text, classification_type, custom_categories)
            results.append(result)
            if progress_callback:
                progress_callback((i + 1) / len(texts), f"Processed {i + 1}/{len(texts)}")
        return results

    def get_available_classifications(self):
        return {
            "available_types": [
                "sentiment", "emotion", "topic", "spam", "toxicity", 
                "language", "ner", "named_entity"
            ],
            "model_info": {
                "sentiment": "Transformer-based sentiment analysis",
                "emotion": "Emotion detection based on sentiment",
                "topic": "Zero-shot topic classification with custom categories",
                "spam": "Rule-based spam detection",
                "toxicity": "Rule-based toxicity detection", 
                "language": "Rule-based language detection",
                "ner": "BERT-based Named Entity Recognition (CoNLL-03)",
                "named_entity": "Advanced BERT NER model"
            }
        }

    def get_performance_stats(self):
        total_time = self.performance_stats["total_processing_time"]
        total_classifications = self.performance_stats["total_classifications"]
        
        return {
            "total_classifications": total_classifications,
            "total_processing_time": round(total_time, 2),
            "average_processing_time": round(total_time / max(1, total_classifications), 3),
            "success_rate": round(
                (total_classifications - self.performance_stats["error_count"]) / max(1, total_classifications) * 100, 2
            ),
            "error_count": self.performance_stats["error_count"]
        }

    def get_model_info(self):
        return {
            "sentiment": "DistilBERT-based sentiment analysis",
            "ner": "BERT-large-cased fine-tuned on CoNLL-03",
            "named_entity": "BERT-base NER model",
            "topic": "BART-large zero-shot classification"
        }

# Create global instance
text_ml_service = TextMLService() 