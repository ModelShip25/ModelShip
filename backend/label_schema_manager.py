"""
Enhanced Label Schema Management System

Provides advanced label schema creation, validation, versioning, and management
for ModelShip projects with support for custom categories and validation rules.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json
import os
import uuid
import logging
from pydantic import BaseModel, Field, validator
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelType(str, Enum):
    """Supported label types"""
    CLASSIFICATION = "classification"           # Single-class classification
    MULTI_CLASSIFICATION = "multi_classification"  # Multi-class classification
    OBJECT_DETECTION = "object_detection"      # Bounding boxes with classes
    SEMANTIC_SEGMENTATION = "semantic_segmentation"  # Pixel-wise labeling
    NAMED_ENTITY = "named_entity"              # Named entity recognition
    TEXT_CLASSIFICATION = "text_classification"  # Text-based classification
    SENTIMENT = "sentiment"                    # Sentiment analysis
    CUSTOM = "custom"                         # Custom label type

class ValidationRule(BaseModel):
    """Validation rule for label schemas"""
    rule_type: str = Field(..., description="Type of validation rule")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    error_message: str = Field(..., description="Error message when validation fails")
    severity: str = Field(default="error", description="Severity: error, warning, info")

class LabelCategory(BaseModel):
    """Individual label category definition"""
    id: str = Field(..., description="Unique identifier for the category")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Detailed description")
    color: Optional[str] = Field(None, description="Display color (hex code)")
    parent_id: Optional[str] = Field(None, description="Parent category for hierarchical labels")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    active: bool = Field(default=True, description="Whether category is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('color')
    def validate_color(cls, v):
        if v and not v.startswith('#') or len(v) != 7:
            raise ValueError('Color must be a valid hex code (e.g., #FF0000)')
        return v

class LabelSchema(BaseModel):
    """Complete label schema definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Schema name")
    description: Optional[str] = Field(None, description="Schema description")
    version: str = Field(default="1.0.0", description="Schema version")
    label_type: LabelType = Field(..., description="Type of labeling")
    project_id: Optional[int] = Field(None, description="Associated project ID")
    
    # Core schema definition
    categories: List[LabelCategory] = Field(..., description="Label categories")
    validation_rules: List[ValidationRule] = Field(default_factory=list)
    
    # Schema metadata
    created_by: Optional[str] = Field(None, description="Creator user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_public: bool = Field(default=False, description="Whether schema is publicly available")
    tags: List[str] = Field(default_factory=list, description="Schema tags")
    
    # Configuration
    allow_custom_labels: bool = Field(default=False, description="Allow users to add custom labels")
    require_confidence: bool = Field(default=True, description="Require confidence scores")
    min_confidence: float = Field(default=0.0, description="Minimum confidence threshold")
    max_labels_per_item: Optional[int] = Field(None, description="Maximum labels per item")
    
    # Auto-approval settings
    auto_approval_enabled: bool = Field(default=True)
    auto_approval_threshold: float = Field(default=0.80, description="Confidence threshold for auto-approval")
    
    class Config:
        use_enum_values = True

class LabelSchemaManager:
    """Advanced label schema management system"""
    
    def __init__(self, schemas_dir: str = "storage/schemas"):
        self.schemas_dir = Path(schemas_dir)
        self.schemas_dir.mkdir(parents=True, exist_ok=True)
        
        # Built-in schema templates
        self.built_in_schemas = self._initialize_built_in_schemas()
        
        # Schema cache for performance
        self._schema_cache = {}
        
        logger.info(f"Label Schema Manager initialized with directory: {self.schemas_dir}")
    
    def _initialize_built_in_schemas(self) -> Dict[str, LabelSchema]:
        """Initialize built-in schema templates"""
        
        schemas = {}
        
        # Image Classification Schema
        schemas["image_classification_general"] = LabelSchema(
            id="image_classification_general",
            name="General Image Classification",
            description="Basic image classification schema with common object categories",
            label_type=LabelType.CLASSIFICATION,
            categories=[
                LabelCategory(id="person", name="Person", description="Human beings", color="#FF6B6B"),
                LabelCategory(id="animal", name="Animal", description="Animals and pets", color="#4ECDC4"),
                LabelCategory(id="vehicle", name="Vehicle", description="Cars, trucks, bikes, etc.", color="#45B7D1"),
                LabelCategory(id="building", name="Building", description="Buildings and structures", color="#96CEB4"),
                LabelCategory(id="nature", name="Nature", description="Natural landscapes, trees, water", color="#FFEAA7"),
                LabelCategory(id="object", name="Object", description="Common objects and items", color="#DDA0DD"),
                LabelCategory(id="other", name="Other", description="Items not fitting other categories", color="#98D8C8")
            ],
            is_public=True,
            auto_approval_threshold=0.85
        )
        
        # Object Detection Schema
        schemas["object_detection_coco"] = LabelSchema(
            id="object_detection_coco",
            name="COCO Object Detection",
            description="Standard COCO dataset categories for object detection",
            label_type=LabelType.OBJECT_DETECTION,
            categories=[
                LabelCategory(id="person", name="Person", color="#FF0000"),
                LabelCategory(id="bicycle", name="Bicycle", color="#00FF00"),
                LabelCategory(id="car", name="Car", color="#0000FF"),
                LabelCategory(id="motorcycle", name="Motorcycle", color="#FFFF00"),
                LabelCategory(id="airplane", name="Airplane", color="#FF00FF"),
                LabelCategory(id="bus", name="Bus", color="#00FFFF"),
                LabelCategory(id="train", name="Train", color="#800000"),
                LabelCategory(id="truck", name="Truck", color="#008000"),
            ],
            is_public=True,
            auto_approval_threshold=0.75
        )
        
        # Text Classification Schema
        schemas["text_sentiment"] = LabelSchema(
            id="text_sentiment",
            name="Sentiment Analysis",
            description="Basic sentiment classification schema",
            label_type=LabelType.SENTIMENT,
            categories=[
                LabelCategory(id="positive", name="Positive", description="Positive sentiment", color="#4CAF50"),
                LabelCategory(id="negative", name="Negative", description="Negative sentiment", color="#F44336"),
                LabelCategory(id="neutral", name="Neutral", description="Neutral sentiment", color="#9E9E9E")
            ],
            is_public=True,
            auto_approval_threshold=0.80
        )
        
        # Named Entity Recognition Schema
        schemas["ner_general"] = LabelSchema(
            id="ner_general",
            name="General Named Entity Recognition",
            description="Standard NER categories for entity extraction",
            label_type=LabelType.NAMED_ENTITY,
            categories=[
                LabelCategory(id="PERSON", name="Person", description="Names of people", color="#FF9800"),
                LabelCategory(id="ORGANIZATION", name="Organization", description="Companies, agencies, institutions", color="#2196F3"),
                LabelCategory(id="LOCATION", name="Location", description="Countries, cities, addresses", color="#4CAF50"),
                LabelCategory(id="MISC", name="Miscellaneous", description="Other named entities", color="#9C27B0")
            ],
            is_public=True,
            auto_approval_threshold=0.70
        )
        
        return schemas
    
    def create_schema(self, schema_data: Dict[str, Any]) -> LabelSchema:
        """Create a new label schema"""
        
        try:
            # Create schema object
            schema = LabelSchema(**schema_data)
            
            # Validate schema
            validation_result = self.validate_schema(schema)
            if not validation_result["is_valid"]:
                raise ValueError(f"Schema validation failed: {validation_result['errors']}")
            
            # Save schema to disk
            schema_path = self.schemas_dir / f"{schema.id}.json"
            with open(schema_path, 'w') as f:
                json.dump(schema.dict(), f, indent=2, default=str)
            
            # Update cache
            self._schema_cache[schema.id] = schema
            
            logger.info(f"Created new schema: {schema.name} (ID: {schema.id})")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            raise
    
    def get_schema(self, schema_id: str) -> Optional[LabelSchema]:
        """Get a schema by ID"""
        
        # Check cache first
        if schema_id in self._schema_cache:
            return self._schema_cache[schema_id]
        
        # Check built-in schemas
        if schema_id in self.built_in_schemas:
            return self.built_in_schemas[schema_id]
        
        # Load from disk
        schema_path = self.schemas_dir / f"{schema_id}.json"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    schema_data = json.load(f)
                
                schema = LabelSchema(**schema_data)
                self._schema_cache[schema_id] = schema
                return schema
                
            except Exception as e:
                logger.error(f"Failed to load schema {schema_id}: {str(e)}")
                return None
        
        return None
    
    def update_schema(self, schema_id: str, updates: Dict[str, Any]) -> Optional[LabelSchema]:
        """Update an existing schema"""
        
        schema = self.get_schema(schema_id)
        if not schema:
            return None
        
        # Don't allow updating built-in schemas
        if schema_id in self.built_in_schemas:
            raise ValueError("Cannot update built-in schemas")
        
        try:
            # Apply updates
            schema_dict = schema.dict()
            schema_dict.update(updates)
            schema_dict["updated_at"] = datetime.utcnow()
            
            # Increment version if schema content changed
            if any(key in updates for key in ["categories", "validation_rules", "label_type"]):
                version_parts = schema.version.split(".")
                version_parts[-1] = str(int(version_parts[-1]) + 1)
                schema_dict["version"] = ".".join(version_parts)
            
            # Create updated schema
            updated_schema = LabelSchema(**schema_dict)
            
            # Validate updated schema
            validation_result = self.validate_schema(updated_schema)
            if not validation_result["is_valid"]:
                raise ValueError(f"Updated schema validation failed: {validation_result['errors']}")
            
            # Save to disk
            schema_path = self.schemas_dir / f"{schema_id}.json"
            with open(schema_path, 'w') as f:
                json.dump(updated_schema.dict(), f, indent=2, default=str)
            
            # Update cache
            self._schema_cache[schema_id] = updated_schema
            
            logger.info(f"Updated schema: {updated_schema.name} (ID: {schema_id})")
            return updated_schema
            
        except Exception as e:
            logger.error(f"Failed to update schema {schema_id}: {str(e)}")
            raise
    
    def delete_schema(self, schema_id: str) -> bool:
        """Delete a schema"""
        
        # Don't allow deleting built-in schemas
        if schema_id in self.built_in_schemas:
            raise ValueError("Cannot delete built-in schemas")
        
        try:
            schema_path = self.schemas_dir / f"{schema_id}.json"
            if schema_path.exists():
                schema_path.unlink()
            
            # Remove from cache
            if schema_id in self._schema_cache:
                del self._schema_cache[schema_id]
            
            logger.info(f"Deleted schema: {schema_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete schema {schema_id}: {str(e)}")
            return False
    
    def list_schemas(self, project_id: Optional[int] = None, include_public: bool = True) -> List[Dict[str, Any]]:
        """List available schemas"""
        
        schemas = []
        
        # Add built-in schemas if including public ones
        if include_public:
            for schema in self.built_in_schemas.values():
                schemas.append({
                    "id": schema.id,
                    "name": schema.name,
                    "description": schema.description,
                    "label_type": schema.label_type,
                    "version": schema.version,
                    "is_built_in": True,
                    "is_public": True,
                    "created_at": schema.created_at,
                    "categories_count": len(schema.categories)
                })
        
        # Add custom schemas
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                
                # Filter by project if specified
                if project_id and schema_data.get("project_id") != project_id:
                    continue
                
                # Include public schemas or project-specific ones
                if not include_public and schema_data.get("is_public", False):
                    continue
                
                schemas.append({
                    "id": schema_data["id"],
                    "name": schema_data["name"],
                    "description": schema_data.get("description"),
                    "label_type": schema_data["label_type"],
                    "version": schema_data["version"],
                    "is_built_in": False,
                    "is_public": schema_data.get("is_public", False),
                    "project_id": schema_data.get("project_id"),
                    "created_at": schema_data["created_at"],
                    "categories_count": len(schema_data.get("categories", []))
                })
                
            except Exception as e:
                logger.error(f"Failed to load schema file {schema_file}: {str(e)}")
                continue
        
        return sorted(schemas, key=lambda x: x["created_at"], reverse=True)
    
    def validate_schema(self, schema: LabelSchema) -> Dict[str, Any]:
        """Validate a label schema"""
        
        errors = []
        warnings = []
        
        # Check for duplicate category IDs
        category_ids = [cat.id for cat in schema.categories]
        if len(category_ids) != len(set(category_ids)):
            errors.append("Duplicate category IDs found")
        
        # Check for empty categories
        if not schema.categories:
            errors.append("Schema must have at least one category")
        
        # Validate category names
        for category in schema.categories:
            if not category.name.strip():
                errors.append(f"Category '{category.id}' has empty name")
            
            # Check color format
            if category.color and not category.color.startswith('#'):
                errors.append(f"Invalid color format for category '{category.id}'")
        
        # Check confidence thresholds
        if schema.min_confidence < 0.0 or schema.min_confidence > 1.0:
            errors.append("Minimum confidence must be between 0.0 and 1.0")
        
        if schema.auto_approval_threshold < 0.0 or schema.auto_approval_threshold > 1.0:
            errors.append("Auto-approval threshold must be between 0.0 and 1.0")
        
        # Validate hierarchical categories
        for category in schema.categories:
            if category.parent_id:
                parent_exists = any(cat.id == category.parent_id for cat in schema.categories)
                if not parent_exists:
                    errors.append(f"Parent category '{category.parent_id}' not found for category '{category.id}'")
        
        # Check for circular references in hierarchy
        if self._has_circular_references(schema.categories):
            errors.append("Circular references detected in category hierarchy")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _has_circular_references(self, categories: List[LabelCategory]) -> bool:
        """Check for circular references in category hierarchy"""
        
        def has_cycle(category_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(category_id)
            rec_stack.add(category_id)
            
            # Find category
            category = next((cat for cat in categories if cat.id == category_id), None)
            if not category or not category.parent_id:
                rec_stack.remove(category_id)
                return False
            
            if category.parent_id in rec_stack:
                return True
            
            if category.parent_id not in visited:
                if has_cycle(category.parent_id, visited, rec_stack):
                    return True
            
            rec_stack.remove(category_id)
            return False
        
        visited = set()
        rec_stack = set()
        
        for category in categories:
            if category.id not in visited:
                if has_cycle(category.id, visited, rec_stack):
                    return True
        
        return False
    
    def get_schema_for_project(self, project_id: int) -> Optional[LabelSchema]:
        """Get the active schema for a specific project"""
        
        # Look for project-specific schema first
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                
                if schema_data.get("project_id") == project_id:
                    return LabelSchema(**schema_data)
                    
            except Exception as e:
                logger.error(f"Failed to load schema file {schema_file}: {str(e)}")
                continue
        
        # Return default schema based on project type (can be enhanced)
        return self.built_in_schemas.get("image_classification_general")
    
    def export_schema(self, schema_id: str, format: str = "json") -> Optional[str]:
        """Export schema in specified format"""
        
        schema = self.get_schema(schema_id)
        if not schema:
            return None
        
        if format.lower() == "json":
            return json.dumps(schema.dict(), indent=2, default=str)
        
        # Add support for other formats (COCO, YOLO, etc.) as needed
        raise ValueError(f"Unsupported export format: {format}")

# Global instance
label_schema_manager = LabelSchemaManager() 