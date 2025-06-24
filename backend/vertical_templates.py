from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float
from database import get_db
from database_base import Base
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class Industry(str, Enum):
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    LEGAL = "legal"
    INDUSTRIAL = "industrial"
    AUTOMOTIVE = "automotive"
    FINANCE = "finance"
    EDUCATION = "education"
    MEDIA = "media"
    ENERGY = "energy"
    CONSTRUCTION = "construction"
    AGRICULTURE = "agriculture"
    GOVERNMENT = "government"
    OTHER = "other"

class VerticalTemplate(Base):
    __tablename__ = "vertical_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    industry = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Schema configuration
    label_schema = Column(JSON, nullable=False)
    confidence_thresholds = Column(JSON)
    validation_rules = Column(JSON)
    
    # Model configuration
    recommended_models = Column(JSON)
    preprocessing_config = Column(JSON)
    
    # Compliance and standards
    compliance_requirements = Column(JSON)
    quality_standards = Column(JSON)
    
    # Usage metadata
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class VerticalConfig:
    name: str
    industry: Industry
    description: str
    label_schema: Dict[str, Any]
    confidence_thresholds: Dict[str, float]
    validation_rules: List[Dict[str, Any]]
    recommended_models: List[str]
    preprocessing_config: Dict[str, Any]
    compliance_requirements: List[str]
    quality_standards: Dict[str, Any]

class VerticalTemplateService:
    def __init__(self):
        self.templates = self._initialize_built_in_templates()
    
    def _initialize_built_in_templates(self) -> Dict[str, VerticalConfig]:
        """Initialize built-in industry templates"""
        
        templates = {}
        
        # Healthcare Template
        templates["healthcare_radiology"] = VerticalConfig(
            name="Healthcare - Radiology",
            industry=Industry.HEALTHCARE,
            description="Medical imaging analysis for radiology departments",
            label_schema={
                "categories": [
                    {"name": "normal", "description": "No abnormalities detected", "color": "#22c55e"},
                    {"name": "abnormal", "description": "Abnormalities present", "color": "#ef4444"},
                    {"name": "inconclusive", "description": "Requires further examination", "color": "#f59e0b"},
                    {"name": "artifact", "description": "Image artifacts present", "color": "#6b7280"},
                    {"name": "fracture", "description": "Bone fracture detected", "color": "#dc2626"},
                    {"name": "tumor", "description": "Potential tumor detected", "color": "#7c2d12"},
                    {"name": "infection", "description": "Signs of infection", "color": "#059669"}
                ],
                "hierarchical": True,
                "multi_label": False,
                "validation_required": True
            },
            confidence_thresholds={
                "auto_approve": 0.95,  # Very high threshold for medical
                "expert_review": 0.80,
                "radiologist_review": 0.60
            },
            validation_rules=[
                {"rule": "medical_license_required", "reviewer_role": "radiologist"},
                {"rule": "dual_review_required", "condition": "abnormal_findings"},
                {"rule": "audit_trail_mandatory", "all_decisions": True}
            ],
            recommended_models=["medical_resnet", "dicom_classifier", "medical_vision_transformer"],
            preprocessing_config={
                "image_normalization": "medical_standard",
                "contrast_enhancement": True,
                "noise_reduction": True,
                "dicom_metadata_extraction": True
            },
            compliance_requirements=["HIPAA", "FDA_510K", "GDPR", "Medical_Device_Regulation"],
            quality_standards={
                "accuracy_requirement": 0.98,
                "sensitivity_requirement": 0.95,
                "specificity_requirement": 0.95,
                "audit_frequency": "daily"
            }
        )
        
        # Retail Template
        templates["retail_product_classification"] = VerticalConfig(
            name="Retail - Product Classification",
            industry=Industry.RETAIL,
            description="E-commerce product categorization and attribute extraction",
            label_schema={
                "categories": [
                    {"name": "clothing", "description": "Apparel and accessories", "color": "#3b82f6"},
                    {"name": "electronics", "description": "Electronic devices and gadgets", "color": "#8b5cf6"},
                    {"name": "home_garden", "description": "Home and garden items", "color": "#10b981"},
                    {"name": "sports", "description": "Sports and outdoor equipment", "color": "#f59e0b"},
                    {"name": "beauty", "description": "Beauty and personal care", "color": "#ec4899"},
                    {"name": "books", "description": "Books and media", "color": "#6b7280"},
                    {"name": "automotive", "description": "Car parts and accessories", "color": "#dc2626"}
                ],
                "attributes": [
                    {"name": "brand", "type": "text", "required": True},
                    {"name": "color", "type": "multi_select", "options": ["red", "blue", "green", "black", "white", "other"]},
                    {"name": "size", "type": "text", "pattern": "^(XS|S|M|L|XL|XXL|\\d+)$"},
                    {"name": "material", "type": "text"},
                    {"name": "price_range", "type": "select", "options": ["budget", "mid_range", "premium", "luxury"]}
                ],
                "hierarchical": True,
                "multi_label": True
            },
            confidence_thresholds={
                "auto_approve": 0.90,
                "human_review": 0.70,
                "quality_check": 0.50
            },
            validation_rules=[
                {"rule": "brand_consistency_check", "field": "brand"},
                {"rule": "category_attribute_matching", "required": True},
                {"rule": "price_range_validation", "field": "price_range"}
            ],
            recommended_models=["retail_classifier", "product_attribute_extractor", "brand_detector"],
            preprocessing_config={
                "background_removal": True,
                "image_standardization": True,
                "text_extraction": "ocr_enabled",
                "logo_detection": True
            },
            compliance_requirements=["GDPR", "CCPA", "Product_Safety_Standards"],
            quality_standards={
                "accuracy_requirement": 0.92,
                "catalog_consistency": 0.98,
                "attribute_completeness": 0.90
            }
        )
        
        # Legal Template
        templates["legal_document_analysis"] = VerticalConfig(
            name="Legal - Document Analysis",
            industry=Industry.LEGAL,
            description="Legal document classification and entity extraction",
            label_schema={
                "categories": [
                    {"name": "contract", "description": "Contractual agreements", "color": "#1e40af"},
                    {"name": "litigation", "description": "Court filings and litigation documents", "color": "#dc2626"},
                    {"name": "compliance", "description": "Regulatory and compliance documents", "color": "#059669"},
                    {"name": "intellectual_property", "description": "Patents, trademarks, copyrights", "color": "#7c2d12"},
                    {"name": "corporate", "description": "Corporate governance documents", "color": "#4f46e5"},
                    {"name": "employment", "description": "HR and employment related", "color": "#0891b2"},
                    {"name": "real_estate", "description": "Property and real estate documents", "color": "#65a30d"}
                ],
                "entities": [
                    {"name": "person", "type": "named_entity", "required": True},
                    {"name": "organization", "type": "named_entity", "required": True},
                    {"name": "date", "type": "temporal", "format": "iso_date"},
                    {"name": "monetary_amount", "type": "financial", "currency": True},
                    {"name": "legal_citation", "type": "legal_reference"},
                    {"name": "jurisdiction", "type": "geographic"},
                    {"name": "contract_term", "type": "duration"}
                ],
                "hierarchical": True,
                "multi_label": False,
                "confidentiality_required": True
            },
            confidence_thresholds={
                "auto_approve": 0.88,
                "attorney_review": 0.75,
                "paralegal_review": 0.60
            },
            validation_rules=[
                {"rule": "attorney_privilege_protection", "all_documents": True},
                {"rule": "client_confidentiality_check", "required": True},
                {"rule": "jurisdiction_validation", "field": "jurisdiction"},
                {"rule": "citation_accuracy_check", "field": "legal_citation"}
            ],
            recommended_models=["legal_bert", "contract_analyzer", "legal_ner"],
            preprocessing_config={
                "text_extraction": "advanced_ocr",
                "page_segmentation": True,
                "redaction_detection": True,
                "legal_formatting_preservation": True
            },
            compliance_requirements=["Attorney_Client_Privilege", "Bar_Association_Rules", "GDPR", "Data_Protection"],
            quality_standards={
                "accuracy_requirement": 0.95,
                "confidentiality_protection": 1.0,
                "audit_trail_mandatory": True,
                "expert_validation_required": True
            }
        )
        
        # Industrial Template
        templates["industrial_quality_control"] = VerticalConfig(
            name="Industrial - Quality Control",
            industry=Industry.INDUSTRIAL,
            description="Manufacturing defect detection and quality assessment",
            label_schema={
                "categories": [
                    {"name": "pass", "description": "Product meets quality standards", "color": "#22c55e"},
                    {"name": "fail", "description": "Product fails quality standards", "color": "#dc2626"},
                    {"name": "defect_surface", "description": "Surface defects detected", "color": "#f59e0b"},
                    {"name": "defect_dimensional", "description": "Dimensional issues", "color": "#ef4444"},
                    {"name": "defect_assembly", "description": "Assembly defects", "color": "#8b5cf6"},
                    {"name": "requires_rework", "description": "Product can be reworked", "color": "#0891b2"},
                    {"name": "scrap", "description": "Product must be scrapped", "color": "#7f1d1d"}
                ],
                "measurements": [
                    {"name": "dimensional_tolerance", "type": "numeric", "unit": "mm", "precision": 3},
                    {"name": "surface_roughness", "type": "numeric", "unit": "Ra", "precision": 2},
                    {"name": "defect_size", "type": "numeric", "unit": "mm2", "precision": 1},
                    {"name": "severity_score", "type": "numeric", "range": [1, 10]}
                ],
                "hierarchical": True,
                "multi_label": True
            },
            confidence_thresholds={
                "auto_approve": 0.95,
                "quality_engineer_review": 0.85,
                "supervisor_review": 0.70
            },
            validation_rules=[
                {"rule": "iso_9001_compliance", "all_products": True},
                {"rule": "statistical_process_control", "sampling_required": True},
                {"rule": "traceability_requirement", "batch_tracking": True}
            ],
            recommended_models=["industrial_vision", "defect_detector", "quality_classifier"],
            preprocessing_config={
                "lighting_normalization": True,
                "perspective_correction": True,
                "measurement_calibration": True,
                "noise_filtering": "industrial_grade"
            },
            compliance_requirements=["ISO_9001", "ISO_14001", "OSHA", "Six_Sigma"],
            quality_standards={
                "accuracy_requirement": 0.96,
                "false_positive_rate": 0.02,
                "false_negative_rate": 0.01,
                "measurement_precision": 0.001
            }
        )
        
        return templates
    
    def get_template(self, template_id: str) -> Optional[VerticalConfig]:
        """Get a specific vertical template"""
        return self.templates.get(template_id)
    
    def list_templates(self, industry: Optional[Industry] = None) -> List[Dict[str, Any]]:
        """List available vertical templates"""
        filtered_templates = []
        
        for template_id, template in self.templates.items():
            if industry is None or template.industry == industry:
                filtered_templates.append({
                    "id": template_id,
                    "name": template.name,
                    "industry": template.industry.value,
                    "description": template.description,
                    "compliance_requirements": template.compliance_requirements,
                    "accuracy_requirement": template.quality_standards.get("accuracy_requirement", 0.9)
                })
        
        return filtered_templates
    
    def create_project_from_template(
        self,
        db: Session,
        template_id: str,
        project_name: str,
        user_id: int,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new project using a vertical template"""
        
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Apply customizations if provided
        project_config = {
            "name": project_name,
            "industry": template.industry.value,
            "label_schema": template.label_schema,
            "confidence_thresholds": template.confidence_thresholds,
            "validation_rules": template.validation_rules,
            "recommended_models": template.recommended_models,
            "preprocessing_config": template.preprocessing_config,
            "compliance_requirements": template.compliance_requirements,
            "quality_standards": template.quality_standards
        }
        
        if customizations:
            project_config.update(customizations)
        
        # Store template usage
        template_record = VerticalTemplate(
            name=template.name,
            industry=template.industry.value,
            description=template.description,
            label_schema=template.label_schema,
            confidence_thresholds=template.confidence_thresholds,
            validation_rules=template.validation_rules,
            recommended_models=template.recommended_models,
            preprocessing_config=template.preprocessing_config,
            compliance_requirements=template.compliance_requirements,
            quality_standards=template.quality_standards
        )
        
        db.add(template_record)
        db.commit()
        db.refresh(template_record)
        
        # Update usage count
        template_record.usage_count += 1
        db.commit()
        
        return {
            "template_id": template_id,
            "template_name": template.name,
            "industry": template.industry.value,
            "project_config": project_config,
            "compliance_ready": True,
            "setup_instructions": self._generate_setup_instructions(template)
        }
    
    def _generate_setup_instructions(self, template: VerticalConfig) -> List[str]:
        """Generate setup instructions for the template"""
        
        instructions = [
            f"1. Configure your project for {template.industry.value} industry standards",
            f"2. Ensure compliance with: {', '.join(template.compliance_requirements)}",
            f"3. Set accuracy requirement to {template.quality_standards.get('accuracy_requirement', 0.9):.1%}",
            f"4. Configure recommended models: {', '.join(template.recommended_models[:3])}"
        ]
        
        if template.industry == Industry.HEALTHCARE:
            instructions.extend([
                "5. Verify all reviewers have medical licenses",
                "6. Enable dual-review for abnormal findings",
                "7. Set up audit trail logging",
                "8. Configure HIPAA-compliant data handling"
            ])
        elif template.industry == Industry.LEGAL:
            instructions.extend([
                "5. Enable attorney-client privilege protection",
                "6. Configure confidentiality settings",
                "7. Set up legal citation validation",
                "8. Enable secure document handling"
            ])
        elif template.industry == Industry.RETAIL:
            instructions.extend([
                "5. Configure brand consistency checking",
                "6. Enable product attribute extraction",
                "7. Set up catalog integration",
                "8. Configure inventory management hooks"
            ])
        elif template.industry == Industry.INDUSTRIAL:
            instructions.extend([
                "5. Configure measurement calibration",
                "6. Enable statistical process control",
                "7. Set up batch traceability",
                "8. Configure quality control workflows"
            ])
        
        return instructions
    
    def validate_compliance(
        self,
        template_id: str,
        project_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate project configuration against compliance requirements"""
        
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        compliance_results = {
            "compliant": True,
            "requirements_checked": len(template.compliance_requirements),
            "passed_checks": [],
            "failed_checks": [],
            "warnings": [],
            "certification_ready": False
        }
        
        # Check each compliance requirement
        for requirement in template.compliance_requirements:
            check_result = self._check_compliance_requirement(requirement, project_config, template)
            
            if check_result["passed"]:
                compliance_results["passed_checks"].append(check_result)
            else:
                compliance_results["failed_checks"].append(check_result)
                compliance_results["compliant"] = False
            
            if check_result.get("warnings"):
                compliance_results["warnings"].extend(check_result["warnings"])
        
        # Overall compliance assessment
        compliance_results["certification_ready"] = (
            compliance_results["compliant"] and 
            len(compliance_results["warnings"]) == 0
        )
        
        return compliance_results
    
    def _check_compliance_requirement(
        self,
        requirement: str,
        project_config: Dict[str, Any],
        template: VerticalConfig
    ) -> Dict[str, Any]:
        """Check a specific compliance requirement"""
        
        check_result = {
            "requirement": requirement,
            "passed": True,
            "details": "",
            "warnings": []
        }
        
        if requirement == "HIPAA":
            # Check HIPAA compliance requirements
            if not project_config.get("encryption_at_rest", False):
                check_result["passed"] = False
                check_result["details"] = "HIPAA requires encryption at rest"
            
            if not project_config.get("audit_logging", False):
                check_result["warnings"].append("Audit logging recommended for HIPAA compliance")
        
        elif requirement == "GDPR":
            # Check GDPR compliance
            if not project_config.get("data_retention_policy", False):
                check_result["passed"] = False
                check_result["details"] = "GDPR requires data retention policy"
            
            if not project_config.get("user_consent_tracking", False):
                check_result["warnings"].append("User consent tracking recommended for GDPR")
        
        elif requirement == "ISO_9001":
            # Check ISO 9001 quality management
            if not project_config.get("quality_control_process", False):
                check_result["passed"] = False
                check_result["details"] = "ISO 9001 requires documented quality control process"
        
        # Add more compliance checks as needed
        
        return check_result

# Create service instance
vertical_service = VerticalTemplateService()

# FastAPI Router
router = APIRouter(prefix="/api/verticals", tags=["vertical_templates"])

@router.get("/templates")
async def list_vertical_templates(
    industry: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List available vertical templates"""
    
    industry_filter = None
    if industry:
        try:
            industry_filter = Industry(industry.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid industry: {industry}")
    
    templates = vertical_service.list_templates(industry_filter)
    
    return {
        "status": "success",
        "industry_filter": industry,
        "total_templates": len(templates),
        "templates": templates,
        "available_industries": [ind.value for ind in Industry]
    }

@router.get("/templates/{template_id}")
async def get_vertical_template(
    template_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific template"""
    
    template = vertical_service.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "status": "success",
        "template_id": template_id,
        "template": {
            "name": template.name,
            "industry": template.industry.value,
            "description": template.description,
            "label_schema": template.label_schema,
            "confidence_thresholds": template.confidence_thresholds,
            "validation_rules": template.validation_rules,
            "recommended_models": template.recommended_models,
            "preprocessing_config": template.preprocessing_config,
            "compliance_requirements": template.compliance_requirements,
            "quality_standards": template.quality_standards
        }
    }

@router.post("/create-project/{template_id}")
async def create_project_from_template(
    template_id: str,
    project_name: str,
    user_id: int = 1,
    customizations: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Create a new project using a vertical template"""
    
    try:
        result = vertical_service.create_project_from_template(
            db, template_id, project_name, user_id, customizations
        )
        return {
            "status": "success",
            "project_created": result
        }
    except Exception as e:
        logger.error(f"Failed to create project from template: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/validate-compliance/{template_id}")
async def validate_compliance(
    template_id: str,
    project_config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Validate project configuration against compliance requirements"""
    
    try:
        compliance_results = vertical_service.validate_compliance(template_id, project_config)
        return {
            "status": "success",
            "template_id": template_id,
            "compliance": compliance_results
        }
    except Exception as e:
        logger.error(f"Compliance validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/industries")
async def get_supported_industries():
    """Get list of supported industries"""
    
    return {
        "status": "success",
        "industries": [
            {
                "code": industry.value,
                "name": industry.value.replace("_", " ").title(),
                "description": f"Templates and compliance for {industry.value} industry"
            }
            for industry in Industry
        ]
    } 