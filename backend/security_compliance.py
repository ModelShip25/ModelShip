from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float
from database import get_db
from database_base import Base
from typing import Dict, List, Any, Optional
import logging
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import jwt
from cryptography.fernet import Fernet
import boto3
from pathlib import Path
import ssl
import os

logger = logging.getLogger(__name__)

class ComplianceStandard(str, Enum):
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOC2 = "soc2"
    CCPA = "ccpa"
    ISO27001 = "iso27001"
    NIST = "nist"
    PCI_DSS = "pci_dss"

class SecurityLevel(str, Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    ENTERPRISE = "enterprise"
    GOVERNMENT = "government"

class AuditEventType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    ADMIN_ACTION = "admin_action"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"

class SecurityAuditLog(Base):
    __tablename__ = "security_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    project_id = Column(Integer, ForeignKey("projects.id"))
    
    # Event details
    event_type = Column(String(50), nullable=False)
    event_description = Column(Text)
    resource_accessed = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Security context
    session_id = Column(String(255))
    risk_score = Column(Float, default=0.0)
    
    # Compliance tracking
    compliance_relevant = Column(Boolean, default=False)
    retention_period_days = Column(Integer, default=2555)  # 7 years for HIPAA
    
    # Metadata
    additional_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ComplianceReport(Base):
    __tablename__ = "compliance_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, nullable=False)
    
    # Report details
    standard = Column(String(50), nullable=False)  # HIPAA, GDPR, etc.
    report_period_start = Column(DateTime, nullable=False)
    report_period_end = Column(DateTime, nullable=False)
    
    # Compliance status
    compliance_score = Column(Float)  # 0-100%
    requirements_met = Column(JSON)   # List of met requirements
    gaps_identified = Column(JSON)    # List of compliance gaps
    risk_assessment = Column(JSON)    # Risk analysis
    
    # Remediation
    remediation_plan = Column(JSON)   # Action items
    next_review_date = Column(DateTime)
    
    # Report metadata
    generated_by = Column(Integer, ForeignKey("users.id"))
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    approved_by = Column(Integer, ForeignKey("users.id"))
    
    status = Column(String(50), default="draft")  # draft, under_review, approved
    created_at = Column(DateTime, default=datetime.utcnow)

class EncryptionKey(Base):
    __tablename__ = "encryption_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, nullable=False)
    
    # Key details
    key_id = Column(String(255), unique=True, nullable=False)
    key_type = Column(String(50), nullable=False)  # customer_managed, platform_managed
    encryption_algorithm = Column(String(50), default="AES-256-GCM")
    
    # Key management
    created_by = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    rotation_schedule_days = Column(Integer, default=90)
    last_rotated = Column(DateTime, default=datetime.utcnow)
    next_rotation = Column(DateTime)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    projects_using = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class ComplianceRequirement:
    id: str
    title: str
    description: str
    category: str
    mandatory: bool
    verification_method: str
    documentation_required: List[str]
    implementation_guidance: str

class SecurityComplianceService:
    def __init__(self):
        self.compliance_frameworks = self._initialize_compliance_frameworks()
        self.encryption_service = EncryptionService()
        security_bearer = HTTPBearer()
    
    def _initialize_compliance_frameworks(self) -> Dict[str, List[ComplianceRequirement]]:
        """Initialize compliance framework requirements"""
        
        frameworks = {}
        
        # HIPAA Requirements
        frameworks[ComplianceStandard.HIPAA.value] = [
            ComplianceRequirement(
                id="hipaa_164_502",
                title="Minimum Necessary Standard",
                description="Covered entities must make reasonable efforts to limit PHI to the minimum necessary",
                category="Privacy Rule",
                mandatory=True,
                verification_method="policy_review",
                documentation_required=["privacy_policy", "access_controls", "audit_logs"],
                implementation_guidance="Implement role-based access controls and data minimization practices"
            ),
            ComplianceRequirement(
                id="hipaa_164_308_a1",
                title="Assigned Security Responsibility",
                description="A covered entity must assign a security officer responsible for security policies",
                category="Administrative Safeguards",
                mandatory=True,
                verification_method="documentation",
                documentation_required=["security_officer_designation", "organizational_chart"],
                implementation_guidance="Designate a qualified security officer and document responsibilities"
            ),
            ComplianceRequirement(
                id="hipaa_164_312_a1",
                title="Access Control",
                description="Assign a unique name and/or number for identifying and tracking user identity",
                category="Technical Safeguards",
                mandatory=True,
                verification_method="technical_review",
                documentation_required=["user_access_matrix", "authentication_logs"],
                implementation_guidance="Implement unique user identification and strong authentication"
            ),
            ComplianceRequirement(
                id="hipaa_164_312_b",
                title="Audit Controls",
                description="Implement hardware, software, and/or procedural mechanisms for audit controls",
                category="Technical Safeguards",
                mandatory=True,
                verification_method="audit_log_review",
                documentation_required=["audit_log_policy", "security_incident_logs"],
                implementation_guidance="Deploy comprehensive audit logging and monitoring systems"
            ),
            ComplianceRequirement(
                id="hipaa_164_312_c1",
                title="Integrity",
                description="PHI must not be improperly altered or destroyed",
                category="Technical Safeguards",
                mandatory=True,
                verification_method="integrity_checks",
                documentation_required=["data_integrity_procedures", "backup_procedures"],
                implementation_guidance="Implement data integrity checks and secure backup procedures"
            )
        ]
        
        # GDPR Requirements
        frameworks[ComplianceStandard.GDPR.value] = [
            ComplianceRequirement(
                id="gdpr_art_6",
                title="Lawfulness of Processing",
                description="Processing shall be lawful only if at least one legal basis applies",
                category="Lawful Basis",
                mandatory=True,
                verification_method="legal_review",
                documentation_required=["legal_basis_assessment", "consent_records"],
                implementation_guidance="Establish and document legal basis for all data processing activities"
            ),
            ComplianceRequirement(
                id="gdpr_art_17",
                title="Right to Erasure",
                description="Data subjects have the right to obtain erasure of personal data",
                category="Individual Rights",
                mandatory=True,
                verification_method="process_review",
                documentation_required=["deletion_procedures", "erasure_logs"],
                implementation_guidance="Implement systematic data deletion capabilities and audit trails"
            ),
            ComplianceRequirement(
                id="gdpr_art_25",
                title="Data Protection by Design",
                description="Implement appropriate technical and organizational measures",
                category="Data Protection",
                mandatory=True,
                verification_method="technical_review",
                documentation_required=["privacy_impact_assessments", "security_measures"],
                implementation_guidance="Integrate privacy considerations into system design and development"
            ),
            ComplianceRequirement(
                id="gdpr_art_32",
                title="Security of Processing",
                description="Implement appropriate technical and organizational measures",
                category="Security",
                mandatory=True,
                verification_method="security_audit",
                documentation_required=["security_policy", "risk_assessments", "incident_response_plan"],
                implementation_guidance="Deploy encryption, access controls, and security monitoring"
            ),
            ComplianceRequirement(
                id="gdpr_art_33",
                title="Breach Notification",
                description="Notify supervisory authority within 72 hours of breach awareness",
                category="Incident Response",
                mandatory=True,
                verification_method="incident_logs",
                documentation_required=["incident_response_procedures", "breach_notification_records"],
                implementation_guidance="Establish rapid breach detection and notification procedures"
            )
        ]
        
        # SOC 2 Requirements
        frameworks[ComplianceStandard.SOC2.value] = [
            ComplianceRequirement(
                id="soc2_cc1_1",
                title="Control Environment",
                description="Demonstrates commitment to integrity and ethical values",
                category="Common Criteria",
                mandatory=True,
                verification_method="policy_review",
                documentation_required=["code_of_conduct", "ethics_policy"],
                implementation_guidance="Establish and enforce ethical standards and conduct policies"
            ),
            ComplianceRequirement(
                id="soc2_cc6_1",
                title="Logical Access Controls",
                description="Logical access security measures restrict access to information",
                category="Common Criteria",
                mandatory=True,
                verification_method="access_review",
                documentation_required=["access_control_policy", "user_access_reviews"],
                implementation_guidance="Implement least privilege access and regular access reviews"
            ),
            ComplianceRequirement(
                id="soc2_cc7_1",
                title="System Operations",
                description="System capacity and monitoring support system requirements",
                category="Common Criteria",
                mandatory=True,
                verification_method="monitoring_review",
                documentation_required=["monitoring_procedures", "capacity_planning"],
                implementation_guidance="Deploy comprehensive system monitoring and capacity management"
            )
        ]
        
        return frameworks
    
    def assess_compliance(
        self,
        db: Session,
        organization_id: int,
        standard: ComplianceStandard,
        project_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assess compliance with a specific standard"""
        
        logger.info(f"Starting compliance assessment for {standard.value}")
        
        requirements = self.compliance_frameworks.get(standard.value, [])
        
        assessment = {
            "standard": standard.value,
            "organization_id": organization_id,
            "assessment_date": datetime.utcnow().isoformat(),
            "total_requirements": len(requirements),
            "requirements_assessment": [],
            "compliance_score": 0.0,
            "gaps_identified": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        met_requirements = 0
        
        for requirement in requirements:
            req_assessment = self._assess_single_requirement(
                db, organization_id, requirement, project_data
            )
            
            assessment["requirements_assessment"].append(req_assessment)
            
            if req_assessment["status"] == "met":
                met_requirements += 1
            elif req_assessment["status"] == "partial":
                met_requirements += 0.5
            else:
                assessment["gaps_identified"].append({
                    "requirement_id": requirement.id,
                    "title": requirement.title,
                    "gap_description": req_assessment.get("gap_description", "Not implemented"),
                    "risk_level": req_assessment.get("risk_level", "medium"),
                    "remediation_effort": req_assessment.get("remediation_effort", "medium")
                })
        
        # Calculate compliance score
        assessment["compliance_score"] = (met_requirements / len(requirements)) * 100 if requirements else 100
        
        # Determine overall risk level
        critical_gaps = len([gap for gap in assessment["gaps_identified"] if gap["risk_level"] == "high"])
        if critical_gaps > 0:
            assessment["risk_level"] = "high"
        elif len(assessment["gaps_identified"]) > len(requirements) * 0.3:
            assessment["risk_level"] = "medium"
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_compliance_recommendations(
            standard, assessment["gaps_identified"]
        )
        
        return assessment
    
    def _assess_single_requirement(
        self,
        db: Session,
        organization_id: int,
        requirement: ComplianceRequirement,
        project_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess a single compliance requirement"""
        
        # This is a simplified assessment - in reality, this would involve
        # checking actual system configurations, policies, and procedures
        
        assessment = {
            "requirement_id": requirement.id,
            "title": requirement.title,
            "category": requirement.category,
            "mandatory": requirement.mandatory,
            "status": "not_assessed",
            "evidence": [],
            "gap_description": "",
            "risk_level": "medium",
            "remediation_effort": "medium"
        }
        
        # Simulate assessment logic based on requirement type
        if "access_control" in requirement.id.lower() or "164_312_a1" in requirement.id:
            # Check if access controls are implemented
            has_rbac = self._check_rbac_implementation(db, organization_id)
            has_mfa = self._check_mfa_enabled(db, organization_id)
            
            if has_rbac and has_mfa:
                assessment["status"] = "met"
                assessment["evidence"] = ["RBAC implemented", "MFA enabled"]
            elif has_rbac or has_mfa:
                assessment["status"] = "partial"
                assessment["gap_description"] = "Partial access control implementation"
            else:
                assessment["status"] = "not_met"
                assessment["gap_description"] = "Access controls not properly implemented"
                assessment["risk_level"] = "high"
        
        elif "audit" in requirement.id.lower() or "164_312_b" in requirement.id:
            # Check audit logging
            has_audit_logs = self._check_audit_logging(db, organization_id)
            
            if has_audit_logs:
                assessment["status"] = "met"
                assessment["evidence"] = ["Comprehensive audit logging enabled"]
            else:
                assessment["status"] = "not_met"
                assessment["gap_description"] = "Audit logging not implemented"
                assessment["risk_level"] = "high"
        
        elif "encryption" in requirement.id.lower() or "security" in requirement.id.lower():
            # Check encryption
            has_encryption = self._check_encryption_implementation(db, organization_id)
            
            if has_encryption:
                assessment["status"] = "met"
                assessment["evidence"] = ["Data encryption implemented"]
            else:
                assessment["status"] = "not_met"
                assessment["gap_description"] = "Data encryption not implemented"
                assessment["risk_level"] = "high"
        
        else:
            # Default assessment for other requirements
            assessment["status"] = "partial"
            assessment["gap_description"] = "Manual review required"
        
        return assessment
    
    def _check_rbac_implementation(self, db: Session, organization_id: int) -> bool:
        """Check if role-based access control is implemented"""
        # This would check actual RBAC configuration
        return True  # Simplified for demo
    
    def _check_mfa_enabled(self, db: Session, organization_id: int) -> bool:
        """Check if multi-factor authentication is enabled"""
        # This would check MFA configuration
        return False  # Simplified for demo
    
    def _check_audit_logging(self, db: Session, organization_id: int) -> bool:
        """Check if audit logging is properly configured"""
        # Check if audit logs exist for the organization
        recent_logs = db.query(SecurityAuditLog).filter(
            SecurityAuditLog.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).limit(1).all()
        
        return len(recent_logs) > 0
    
    def _check_encryption_implementation(self, db: Session, organization_id: int) -> bool:
        """Check if encryption is properly implemented"""
        # Check if encryption keys are configured
        encryption_keys = db.query(EncryptionKey).filter(
            EncryptionKey.organization_id == organization_id,
            EncryptionKey.is_active == True
        ).limit(1).all()
        
        return len(encryption_keys) > 0
    
    def _generate_compliance_recommendations(
        self,
        standard: ComplianceStandard,
        gaps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on compliance gaps"""
        
        recommendations = []
        
        for gap in gaps:
            if "access_control" in gap["requirement_id"].lower():
                recommendations.append({
                    "priority": "high",
                    "title": "Implement Role-Based Access Control",
                    "description": "Deploy comprehensive RBAC system with least privilege principles",
                    "implementation_steps": [
                        "Define user roles and permissions matrix",
                        "Configure RBAC in application",
                        "Enable multi-factor authentication",
                        "Implement regular access reviews"
                    ],
                    "estimated_effort": "2-4 weeks",
                    "compliance_impact": "High"
                })
            
            elif "audit" in gap["requirement_id"].lower():
                recommendations.append({
                    "priority": "high",
                    "title": "Deploy Comprehensive Audit Logging",
                    "description": "Implement detailed audit logging for all system activities",
                    "implementation_steps": [
                        "Configure audit log collection",
                        "Set up log retention policies",
                        "Deploy log monitoring and alerting",
                        "Create audit report automation"
                    ],
                    "estimated_effort": "1-2 weeks",
                    "compliance_impact": "High"
                })
            
            elif "encryption" in gap["requirement_id"].lower():
                recommendations.append({
                    "priority": "critical",
                    "title": "Implement Data Encryption",
                    "description": "Deploy encryption for data at rest and in transit",
                    "implementation_steps": [
                        "Configure database encryption",
                        "Implement API encryption (TLS)",
                        "Set up key management system",
                        "Encrypt file storage"
                    ],
                    "estimated_effort": "2-3 weeks",
                    "compliance_impact": "Critical"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "medium",
            "title": "Regular Compliance Monitoring",
            "description": "Establish ongoing compliance monitoring and reporting",
            "implementation_steps": [
                "Schedule regular compliance assessments",
                "Set up automated compliance monitoring",
                "Create compliance dashboard",
                "Establish compliance training program"
            ],
            "estimated_effort": "1-2 weeks",
            "compliance_impact": "Medium"
        })
        
        return recommendations
    
    def log_security_event(
        self,
        db: Session,
        user_id: Optional[int],
        event_type: AuditEventType,
        description: str,
        project_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        risk_score: float = 0.0
    ) -> int:
        """Log a security event for audit purposes"""
        
        audit_log = SecurityAuditLog(
            user_id=user_id,
            project_id=project_id,
            event_type=event_type.value,
            event_description=description,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score,
            compliance_relevant=self._is_compliance_relevant(event_type),
            additional_data=additional_data or {}
        )
        
        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)
        
        # Check if this event triggers any security alerts
        if risk_score > 0.7 or event_type == AuditEventType.SECURITY_EVENT:
            self._trigger_security_alert(db, audit_log)
        
        return audit_log.id
    
    def _is_compliance_relevant(self, event_type: AuditEventType) -> bool:
        """Determine if an event is relevant for compliance reporting"""
        
        compliance_relevant_events = [
            AuditEventType.DATA_ACCESS,
            AuditEventType.DATA_EXPORT,
            AuditEventType.ADMIN_ACTION,
            AuditEventType.SECURITY_EVENT
        ]
        
        return event_type in compliance_relevant_events
    
    def _trigger_security_alert(self, db: Session, audit_log: SecurityAuditLog):
        """Trigger security alert for high-risk events"""
        
        logger.warning(f"Security alert triggered for event {audit_log.id}: {audit_log.event_description}")
        
        # In a real implementation, this would:
        # - Send notifications to security team
        # - Create incident tickets
        # - Trigger automated response procedures
    
    def generate_compliance_report(
        self,
        db: Session,
        organization_id: int,
        standard: ComplianceStandard,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        logger.info(f"Generating {standard.value} compliance report for organization {organization_id}")
        
        # Perform compliance assessment
        assessment = self.assess_compliance(db, organization_id, standard)
        
        # Get relevant audit logs
        audit_logs = db.query(SecurityAuditLog).filter(
            SecurityAuditLog.timestamp >= start_date,
            SecurityAuditLog.timestamp <= end_date,
            SecurityAuditLog.compliance_relevant == True
        ).all()
        
        # Generate report
        report_data = {
            "organization_id": organization_id,
            "standard": standard.value,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "compliance_assessment": assessment,
            "audit_summary": {
                "total_events": len(audit_logs),
                "high_risk_events": len([log for log in audit_logs if log.risk_score > 0.7]),
                "data_access_events": len([log for log in audit_logs if log.event_type == AuditEventType.DATA_ACCESS.value]),
                "security_events": len([log for log in audit_logs if log.event_type == AuditEventType.SECURITY_EVENT.value])
            },
            "recommendations": assessment["recommendations"],
            "next_steps": self._get_compliance_next_steps(standard, assessment),
            "certification_readiness": assessment["compliance_score"] >= 85
        }
        
        # Save report to database
        report = ComplianceReport(
            organization_id=organization_id,
            standard=standard.value,
            report_period_start=start_date,
            report_period_end=end_date,
            compliance_score=assessment["compliance_score"],
            requirements_met=assessment["requirements_assessment"],
            gaps_identified=assessment["gaps_identified"],
            risk_assessment={"overall_risk": assessment["risk_level"]},
            remediation_plan=assessment["recommendations"]
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        report_data["report_id"] = report.id
        
        return report_data
    
    def _get_compliance_next_steps(
        self,
        standard: ComplianceStandard,
        assessment: Dict[str, Any]
    ) -> List[str]:
        """Get next steps for compliance improvement"""
        
        next_steps = []
        
        if assessment["compliance_score"] < 50:
            next_steps.append("Focus on critical security controls implementation")
            next_steps.append("Conduct gap analysis with compliance consultant")
        elif assessment["compliance_score"] < 85:
            next_steps.append("Address remaining compliance gaps")
            next_steps.append("Prepare for third-party compliance audit")
        else:
            next_steps.append("Schedule third-party compliance certification")
            next_steps.append("Maintain ongoing compliance monitoring")
        
        if standard == ComplianceStandard.HIPAA:
            next_steps.append("Conduct Business Associate Agreement review")
            next_steps.append("Schedule HIPAA risk assessment")
        elif standard == ComplianceStandard.GDPR:
            next_steps.append("Review Data Protection Impact Assessments")
            next_steps.append("Update privacy notices and consent mechanisms")
        elif standard == ComplianceStandard.SOC2:
            next_steps.append("Engage SOC 2 auditor for Type II examination")
            next_steps.append("Implement continuous monitoring controls")
        
        return next_steps

class EncryptionService:
    """Service for managing encryption keys and data protection"""
    
    def __init__(self):
        self.key_store = {}  # In production, this would be AWS KMS, HashiCorp Vault, etc.
    
    def create_encryption_key(
        self,
        db: Session,
        organization_id: int,
        key_type: str = "customer_managed",
        created_by: int = None
    ) -> str:
        """Create new encryption key"""
        
        # Generate key
        key = Fernet.generate_key()
        key_id = f"key_{organization_id}_{secrets.token_hex(8)}"
        
        # Store key (in production, use proper key management service)
        self.key_store[key_id] = key
        
        # Save key metadata to database
        encryption_key = EncryptionKey(
            organization_id=organization_id,
            key_id=key_id,
            key_type=key_type,
            created_by=created_by,
            next_rotation=datetime.utcnow() + timedelta(days=90)
        )
        
        db.add(encryption_key)
        db.commit()
        
        logger.info(f"Created encryption key {key_id} for organization {organization_id}")
        
        return key_id
    
    def encrypt_data(self, data: str, key_id: str) -> str:
        """Encrypt data using specified key"""
        
        if key_id not in self.key_store:
            raise ValueError(f"Encryption key {key_id} not found")
        
        fernet = Fernet(self.key_store[key_id])
        encrypted_data = fernet.encrypt(data.encode())
        
        return encrypted_data.decode()
    
    def decrypt_data(self, encrypted_data: str, key_id: str) -> str:
        """Decrypt data using specified key"""
        
        if key_id not in self.key_store:
            raise ValueError(f"Encryption key {key_id} not found")
        
        fernet = Fernet(self.key_store[key_id])
        decrypted_data = fernet.decrypt(encrypted_data.encode())
        
        return decrypted_data.decode()

# Create service instances
security_service = SecurityComplianceService()
encryption_service = EncryptionService()

# FastAPI Router
router = APIRouter(prefix="/api/security-compliance", tags=["security_compliance"])

@router.post("/assess-compliance")
async def assess_compliance(
    organization_id: int,
    standard: str,
    project_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Assess compliance with specific standard"""
    
    try:
        compliance_standard = ComplianceStandard(standard.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid compliance standard: {standard}")
    
    try:
        assessment = security_service.assess_compliance(
            db, organization_id, compliance_standard, project_data
        )
        
        return {
            "status": "success",
            "assessment": assessment
        }
        
    except Exception as e:
        logger.error(f"Compliance assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-report")
async def generate_compliance_report(
    organization_id: int,
    standard: str,
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    """Generate compliance report"""
    
    try:
        compliance_standard = ComplianceStandard(standard.lower())
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {e}")
    
    try:
        report = security_service.generate_compliance_report(
            db, organization_id, compliance_standard, start_dt, end_dt
        )
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log-event")
async def log_security_event(
    event_type: str,
    description: str,
    user_id: Optional[int] = None,
    project_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    risk_score: float = 0.0,
    additional_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Log security event for audit purposes"""
    
    try:
        audit_event_type = AuditEventType(event_type.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
    
    try:
        log_id = security_service.log_security_event(
            db=db,
            user_id=user_id,
            event_type=audit_event_type,
            description=description,
            project_id=project_id,
            ip_address=ip_address,
            user_agent=user_agent,
            additional_data=additional_data,
            risk_score=risk_score
        )
        
        return {
            "status": "success",
            "log_id": log_id
        }
        
    except Exception as e:
        logger.error(f"Event logging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit-logs")
async def get_audit_logs(
    organization_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get security audit logs"""
    
    query = db.query(SecurityAuditLog)
    
    if start_date:
        start_dt = datetime.fromisoformat(start_date)
        query = query.filter(SecurityAuditLog.timestamp >= start_dt)
    
    if end_date:
        end_dt = datetime.fromisoformat(end_date)
        query = query.filter(SecurityAuditLog.timestamp <= end_dt)
    
    if event_type:
        query = query.filter(SecurityAuditLog.event_type == event_type)
    
    logs = query.order_by(SecurityAuditLog.timestamp.desc()).limit(limit).all()
    
    return {
        "status": "success",
        "total_logs": len(logs),
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "event_type": log.event_type,
                "description": log.event_description,
                "timestamp": log.timestamp.isoformat(),
                "risk_score": log.risk_score,
                "ip_address": log.ip_address
            }
            for log in logs
        ]
    }

@router.post("/create-encryption-key")
async def create_encryption_key(
    organization_id: int,
    key_type: str = "customer_managed",
    created_by: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Create new encryption key"""
    
    try:
        key_id = encryption_service.create_encryption_key(
            db, organization_id, key_type, created_by
        )
        
        return {
            "status": "success",
            "key_id": key_id,
            "message": "Encryption key created successfully"
        }
        
    except Exception as e:
        logger.error(f"Key creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/compliance-standards")
async def get_supported_compliance_standards():
    """Get list of supported compliance standards"""
    
    return {
        "status": "success",
        "standards": [
            {
                "code": standard.value,
                "name": standard.value.upper(),
                "description": f"Compliance framework for {standard.value.upper()}",
                "industry": self._get_standard_industry(standard)
            }
            for standard in ComplianceStandard
        ]
    }

def _get_standard_industry(standard: ComplianceStandard) -> str:
    """Get primary industry for compliance standard"""
    
    industry_mapping = {
        ComplianceStandard.HIPAA: "Healthcare",
        ComplianceStandard.GDPR: "General (EU)",
        ComplianceStandard.SOC2: "Technology",
        ComplianceStandard.CCPA: "General (California)",
        ComplianceStandard.ISO27001: "General",
        ComplianceStandard.NIST: "Government",
        ComplianceStandard.PCI_DSS: "Financial"
    }
    
    return industry_mapping.get(standard, "General") 