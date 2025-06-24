from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float, Enum as SQLEnum
from database import get_db
from database_base import Base
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Email functionality optional
    pass

logger = logging.getLogger(__name__)

class ExpertLevel(str, Enum):
    SPECIALIST = "specialist"  # Domain experts (e.g., radiologists, legal associates)
    CONSULTANT = "consultant"  # Senior experts with extensive experience
    CERTIFIED = "certified"    # Certified professionals with credentials
    ACADEMIC = "academic"      # Academic researchers and professors
    AGRICULTURAL = "agricultural"  # Agricultural experts
    ENERGY = "energy"  # Energy experts
    CONSTRUCTION = "construction"  # Construction experts
    GOVERNMENT = "government"  # Government experts
    OTHER = "other"  # Other experts

class ExpertStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ON_VACATION = "on_vacation"

class ReviewUrgency(str, Enum):
    LOW = "low"           # 72 hours
    NORMAL = "normal"     # 24 hours
    HIGH = "high"         # 8 hours
    CRITICAL = "critical" # 2 hours

class Expert(Base):
    __tablename__ = "experts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Professional information
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    title = Column(String(255))  # Dr., Prof., Esq., etc.
    organization = Column(String(255))
    
    # Expertise
    expertise_domains = Column(JSON, nullable=False)  # ["radiology", "cardiology"]
    industries = Column(JSON, nullable=False)  # ["healthcare", "legal"]
    certifications = Column(JSON)  # Professional certifications
    languages = Column(JSON, default=lambda: ["English"])
    
    # Professional details
    years_experience = Column(Integer)
    education = Column(JSON)  # Degrees and institutions
    license_numbers = Column(JSON)  # Professional license info
    cv_url = Column(String(500))
    linkedin_url = Column(String(500))
    
    # Platform details
    expert_level = Column(String(50), nullable=False)
    hourly_rate = Column(Float)  # USD per hour
    availability_hours = Column(JSON)  # Weekly availability schedule
    time_zone = Column(String(50))
    status = Column(String(50), default=ExpertStatus.AVAILABLE.value)
    
    # Performance metrics
    reviews_completed = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    average_turnaround_hours = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    
    # Platform metadata
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime)

class ExpertReviewRequest(Base):
    __tablename__ = "expert_review_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    expert_id = Column(Integer, ForeignKey("experts.id"), nullable=False)
    
    # Request details
    title = Column(String(255), nullable=False)
    description = Column(Text)
    items_to_review = Column(JSON, nullable=False)  # List of item IDs
    urgency = Column(String(50), default=ReviewUrgency.NORMAL.value)
    
    # Requirements
    required_expertise = Column(JSON, nullable=False)
    estimated_hours = Column(Float)
    max_budget = Column(Float)  # USD
    deadline = Column(DateTime)
    
    # Status tracking
    status = Column(String(50), default="pending")  # pending, accepted, in_progress, completed, cancelled
    accepted_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Results
    expert_feedback = Column(Text)
    expert_rating = Column(Integer)  # 1-5 rating of the task
    reviewed_items = Column(JSON)  # Results from expert review
    
    # Payment
    quoted_rate = Column(Float)
    actual_hours = Column(Float)
    total_cost = Column(Float)
    paid_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class ExpertMatchCriteria:
    domains: List[str]
    industries: List[str]
    min_experience: int = 0
    min_rating: float = 4.0
    max_hourly_rate: Optional[float] = None
    urgency: ReviewUrgency = ReviewUrgency.NORMAL
    languages: List[str] = None

class ExpertInLoopService:
    def __init__(self):
        self.expert_pool = self._initialize_expert_pool()
    
    def _initialize_expert_pool(self) -> List[Dict[str, Any]]:
        """Initialize built-in expert pool"""
        
        experts = [
            # Healthcare Experts
            {
                "name": "Dr. Sarah Chen",
                "email": "s.chen@medicalexperts.com",
                "title": "Dr.",
                "organization": "Johns Hopkins Medical Center",
                "expertise_domains": ["radiology", "cardiology", "emergency_medicine"],
                "industries": ["healthcare"],
                "certifications": ["Board Certified Radiologist", "ABIM Internal Medicine"],
                "languages": ["English", "Mandarin"],
                "years_experience": 12,
                "education": [
                    {"degree": "MD", "institution": "Harvard Medical School", "year": 2012},
                    {"degree": "Residency", "institution": "Mayo Clinic", "year": 2016}
                ],
                "license_numbers": [{"state": "MD", "number": "D12345", "expires": "2025-12-31"}],
                "expert_level": ExpertLevel.CONSULTANT.value,
                "hourly_rate": 450.0,
                "time_zone": "America/New_York",
                "availability_hours": {
                    "monday": ["09:00", "17:00"],
                    "tuesday": ["09:00", "17:00"],
                    "wednesday": ["09:00", "17:00"],
                    "thursday": ["09:00", "17:00"],
                    "friday": ["09:00", "15:00"]
                },
                "average_rating": 4.9,
                "reviews_completed": 287,
                "accuracy_score": 0.97
            },
            {
                "name": "Dr. Michael Rodriguez",
                "email": "m.rodriguez@pathologyexperts.com",
                "title": "Dr.",
                "organization": "Stanford Medical Center",
                "expertise_domains": ["pathology", "oncology", "hematology"],
                "industries": ["healthcare"],
                "certifications": ["Board Certified Pathologist", "Molecular Pathology Fellowship"],
                "languages": ["English", "Spanish"],
                "years_experience": 15,
                "education": [
                    {"degree": "MD", "institution": "Stanford University", "year": 2009},
                    {"degree": "PhD", "institution": "Stanford University", "year": 2009}
                ],
                "expert_level": ExpertLevel.CONSULTANT.value,
                "hourly_rate": 500.0,
                "time_zone": "America/Los_Angeles",
                "average_rating": 4.8,
                "reviews_completed": 156,
                "accuracy_score": 0.98
            },
            
            # Legal Experts
            {
                "name": "Sarah Johnson, Esq.",
                "email": "s.johnson@legalexperts.com",
                "title": "Esq.",
                "organization": "Davis Polk & Wardwell LLP",
                "expertise_domains": ["contract_law", "corporate_law", "securities_law"],
                "industries": ["legal", "finance"],
                "certifications": ["NY Bar", "Securities Law Certification"],
                "languages": ["English", "French"],
                "years_experience": 18,
                "education": [
                    {"degree": "JD", "institution": "Harvard Law School", "year": 2006},
                    {"degree": "LLM", "institution": "Cambridge University", "year": 2007}
                ],
                "license_numbers": [{"state": "NY", "number": "L56789", "expires": "2025-12-31"}],
                "expert_level": ExpertLevel.CONSULTANT.value,
                "hourly_rate": 850.0,
                "time_zone": "America/New_York",
                "average_rating": 4.9,
                "reviews_completed": 198,
                "accuracy_score": 0.96
            },
            {
                "name": "Robert Kim, Esq.",
                "email": "r.kim@ipexperts.com",
                "title": "Esq.",
                "organization": "Wilson Sonsini Goodrich & Rosati",
                "expertise_domains": ["intellectual_property", "patent_law", "technology_law"],
                "industries": ["legal", "technology"],
                "certifications": ["CA Bar", "USPTO Registered Patent Attorney"],
                "languages": ["English", "Korean"],
                "years_experience": 10,
                "education": [
                    {"degree": "JD", "institution": "UC Berkeley Law", "year": 2014},
                    {"degree": "BS", "institution": "MIT", "year": 2011, "field": "Computer Science"}
                ],
                "expert_level": ExpertLevel.SPECIALIST.value,
                "hourly_rate": 650.0,
                "time_zone": "America/Los_Angeles",
                "average_rating": 4.7,
                "reviews_completed": 134,
                "accuracy_score": 0.94
            },
            
            # Industrial Experts
            {
                "name": "Dr. Jennifer Walsh",
                "email": "j.walsh@qualityexperts.com",
                "title": "Dr.",
                "organization": "Quality Consulting International",
                "expertise_domains": ["quality_control", "six_sigma", "iso_standards", "manufacturing"],
                "industries": ["industrial", "automotive", "aerospace"],
                "certifications": ["Six Sigma Black Belt", "ISO 9001 Lead Auditor", "ASQ CQE"],
                "languages": ["English", "German"],
                "years_experience": 20,
                "education": [
                    {"degree": "PhD", "institution": "Georgia Tech", "year": 2004, "field": "Industrial Engineering"},
                    {"degree": "MS", "institution": "Purdue University", "year": 2000}
                ],
                "expert_level": ExpertLevel.CONSULTANT.value,
                "hourly_rate": 350.0,
                "time_zone": "America/Chicago",
                "average_rating": 4.8,
                "reviews_completed": 312,
                "accuracy_score": 0.95
            },
            
            # Retail Experts
            {
                "name": "Maria Gonzalez",
                "email": "m.gonzalez@retailexperts.com",
                "title": "Ms.",
                "organization": "Retail Analytics Consulting",
                "expertise_domains": ["retail_analytics", "product_categorization", "brand_management"],
                "industries": ["retail", "fashion", "consumer_goods"],
                "certifications": ["Certified Retail Analytics Professional", "Google Analytics Certified"],
                "languages": ["English", "Spanish", "Portuguese"],
                "years_experience": 12,
                "education": [
                    {"degree": "MBA", "institution": "Wharton School", "year": 2012},
                    {"degree": "BS", "institution": "NYU Stern", "year": 2010, "field": "Marketing"}
                ],
                "expert_level": ExpertLevel.SPECIALIST.value,
                "hourly_rate": 275.0,
                "time_zone": "America/New_York",
                "average_rating": 4.6,
                "reviews_completed": 203,
                "accuracy_score": 0.92
            }
        ]
        
        return experts
    
    def find_matching_experts(
        self,
        criteria: ExpertMatchCriteria,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find experts matching the specified criteria"""
        
        matching_experts = []
        
        for expert in self.expert_pool:
            # Check domain expertise
            domain_match = any(domain in expert["expertise_domains"] for domain in criteria.domains)
            if not domain_match:
                continue
            
            # Check industry expertise
            industry_match = any(industry in expert["industries"] for industry in criteria.industries)
            if not industry_match:
                continue
            
            # Check experience
            if expert["years_experience"] < criteria.min_experience:
                continue
            
            # Check rating
            if expert["average_rating"] < criteria.min_rating:
                continue
            
            # Check hourly rate
            if criteria.max_hourly_rate and expert["hourly_rate"] > criteria.max_hourly_rate:
                continue
            
            # Check language requirements
            if criteria.languages:
                language_match = any(lang in expert["languages"] for lang in criteria.languages)
                if not language_match:
                    continue
            
            # Calculate match score
            match_score = self._calculate_match_score(expert, criteria)
            expert_with_score = expert.copy()
            expert_with_score["match_score"] = match_score
            
            matching_experts.append(expert_with_score)
        
        # Sort by match score and return top matches
        matching_experts.sort(key=lambda x: x["match_score"], reverse=True)
        return matching_experts[:limit]
    
    def _calculate_match_score(
        self,
        expert: Dict[str, Any],
        criteria: ExpertMatchCriteria
    ) -> float:
        """Calculate how well an expert matches the criteria"""
        
        score = 0.0
        
        # Domain expertise match (40% weight)
        domain_matches = sum(1 for domain in criteria.domains if domain in expert["expertise_domains"])
        domain_score = (domain_matches / len(criteria.domains)) * 0.4
        score += domain_score
        
        # Rating score (25% weight)
        rating_score = (expert["average_rating"] / 5.0) * 0.25
        score += rating_score
        
        # Experience score (20% weight)
        experience_score = min(expert["years_experience"] / 20.0, 1.0) * 0.20
        score += experience_score
        
        # Accuracy score (15% weight)
        accuracy_score = expert["accuracy_score"] * 0.15
        score += accuracy_score
        
        return score
    
    def create_review_request(
        self,
        db: Session,
        project_id: int,
        title: str,
        description: str,
        items_to_review: List[int],
        required_expertise: List[str],
        urgency: ReviewUrgency = ReviewUrgency.NORMAL,
        max_budget: Optional[float] = None,
        estimated_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a new expert review request"""
        
        # Calculate deadline based on urgency
        urgency_hours = {
            ReviewUrgency.LOW: 72,
            ReviewUrgency.NORMAL: 24,
            ReviewUrgency.HIGH: 8,
            ReviewUrgency.CRITICAL: 2
        }
        
        deadline = datetime.utcnow() + timedelta(hours=urgency_hours[urgency])
        
        # Find matching experts
        criteria = ExpertMatchCriteria(
            domains=required_expertise,
            industries=["healthcare"],  # Would be determined from project
            max_hourly_rate=max_budget / estimated_hours if max_budget and estimated_hours else None,
            urgency=urgency
        )
        
        matching_experts = self.find_matching_experts(criteria)
        
        # Create the request
        request = ExpertReviewRequest(
            project_id=project_id,
            expert_id=matching_experts[0]["id"] if matching_experts else None,
            title=title,
            description=description,
            items_to_review=items_to_review,
            urgency=urgency.value,
            required_expertise=required_expertise,
            estimated_hours=estimated_hours,
            max_budget=max_budget,
            deadline=deadline
        )
        
        db.add(request)
        db.commit()
        db.refresh(request)
        
        # Notify matching experts
        self._notify_experts(matching_experts[:3], request)
        
        return {
            "request_id": request.id,
            "deadline": deadline.isoformat(),
            "matching_experts": len(matching_experts),
            "top_matches": matching_experts[:3],
            "estimated_cost": self._estimate_cost(matching_experts[:3], estimated_hours)
        }
    
    def _notify_experts(
        self,
        experts: List[Dict[str, Any]],
        request: ExpertReviewRequest
    ):
        """Notify experts about new review request"""
        
        for expert in experts:
            # In a real implementation, this would send actual emails
            logger.info(f"Notifying expert {expert['name']} about review request {request.id}")
            
            # Email content would include:
            # - Request details
            # - Estimated time and compensation
            # - Deadline
            # - Link to accept/decline
    
    def _estimate_cost(
        self,
        experts: List[Dict[str, Any]],
        estimated_hours: Optional[float]
    ) -> Dict[str, float]:
        """Estimate costs for the review"""
        
        if not experts or not estimated_hours:
            return {"min": 0, "max": 0, "average": 0}
        
        rates = [expert["hourly_rate"] for expert in experts]
        
        return {
            "min": min(rates) * estimated_hours,
            "max": max(rates) * estimated_hours,
            "average": (sum(rates) / len(rates)) * estimated_hours
        }
    
    def accept_review_request(
        self,
        db: Session,
        request_id: int,
        expert_id: int,
        quoted_rate: float
    ) -> Dict[str, Any]:
        """Expert accepts a review request"""
        
        request = db.query(ExpertReviewRequest).filter(
            ExpertReviewRequest.id == request_id
        ).first()
        
        if not request:
            raise ValueError("Review request not found")
        
        if request.status != "pending":
            raise ValueError("Request is no longer available")
        
        # Update request
        request.expert_id = expert_id
        request.status = "accepted"
        request.accepted_at = datetime.utcnow()
        request.quoted_rate = quoted_rate
        
        db.commit()
        
        return {
            "status": "accepted",
            "expert_assigned": expert_id,
            "quoted_rate": quoted_rate,
            "deadline": request.deadline.isoformat()
        }
    
    def submit_expert_review(
        self,
        db: Session,
        request_id: int,
        reviewed_items: List[Dict[str, Any]],
        expert_feedback: str,
        actual_hours: float,
        expert_rating: int
    ) -> Dict[str, Any]:
        """Expert submits completed review"""
        
        request = db.query(ExpertReviewRequest).filter(
            ExpertReviewRequest.id == request_id
        ).first()
        
        if not request:
            raise ValueError("Review request not found")
        
        if request.status != "accepted" and request.status != "in_progress":
            raise ValueError("Invalid request status for submission")
        
        # Update request
        request.status = "completed"
        request.completed_at = datetime.utcnow()
        request.reviewed_items = reviewed_items
        request.expert_feedback = expert_feedback
        request.actual_hours = actual_hours
        request.expert_rating = expert_rating
        request.total_cost = request.quoted_rate * actual_hours
        
        db.commit()
        
        # Update expert performance metrics
        self._update_expert_metrics(db, request.expert_id, request)
        
        return {
            "status": "completed",
            "items_reviewed": len(reviewed_items),
            "total_cost": request.total_cost,
            "expert_feedback": expert_feedback
        }
    
    def _update_expert_metrics(
        self,
        db: Session,
        expert_id: int,
        completed_request: ExpertReviewRequest
    ):
        """Update expert performance metrics"""
        
        # In a real implementation, this would update the Expert table
        logger.info(f"Updating metrics for expert {expert_id}")
        
        # Calculate turnaround time
        turnaround_hours = (
            completed_request.completed_at - completed_request.accepted_at
        ).total_seconds() / 3600
        
        # Update reviews completed, average rating, turnaround time, etc.

# Create service instance
expert_service = ExpertInLoopService()

# FastAPI Router
router = APIRouter(prefix="/api/experts", tags=["expert_in_loop"])

@router.get("/available")
async def get_available_experts(
    domains: Optional[str] = None,
    industries: Optional[str] = None,
    max_rate: Optional[float] = None,
    min_rating: Optional[float] = 4.0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get available experts matching criteria"""
    
    criteria = ExpertMatchCriteria(
        domains=domains.split(",") if domains else [],
        industries=industries.split(",") if industries else [],
        max_hourly_rate=max_rate,
        min_rating=min_rating
    )
    
    experts = expert_service.find_matching_experts(criteria, limit)
    
    return {
        "status": "success",
        "criteria": {
            "domains": criteria.domains,
            "industries": criteria.industries,
            "max_rate": max_rate,
            "min_rating": min_rating
        },
        "total_found": len(experts),
        "experts": experts
    }

@router.post("/request-review")
async def request_expert_review(
    project_id: int,
    title: str,
    description: str,
    items_to_review: List[int],
    required_expertise: List[str],
    urgency: str = "normal",
    max_budget: Optional[float] = None,
    estimated_hours: Optional[float] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Request expert review for specific items"""
    
    try:
        urgency_level = ReviewUrgency(urgency.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid urgency level: {urgency}")
    
    try:
        result = expert_service.create_review_request(
            db=db,
            project_id=project_id,
            title=title,
            description=description,
            items_to_review=items_to_review,
            required_expertise=required_expertise,
            urgency=urgency_level,
            max_budget=max_budget,
            estimated_hours=estimated_hours
        )
        
        return {
            "status": "success",
            "request": result
        }
        
    except Exception as e:
        logger.error(f"Failed to create expert review request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/accept-request/{request_id}")
async def accept_expert_request(
    request_id: int,
    expert_id: int,
    quoted_rate: float,
    db: Session = Depends(get_db)
):
    """Expert accepts a review request"""
    
    try:
        result = expert_service.accept_review_request(
            db=db,
            request_id=request_id,
            expert_id=expert_id,
            quoted_rate=quoted_rate
        )
        
        return {
            "status": "success",
            "accepted": result
        }
        
    except Exception as e:
        logger.error(f"Failed to accept review request: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/submit-review/{request_id}")
async def submit_expert_review(
    request_id: int,
    reviewed_items: List[Dict[str, Any]],
    expert_feedback: str,
    actual_hours: float,
    expert_rating: int,
    db: Session = Depends(get_db)
):
    """Expert submits completed review"""
    
    if expert_rating < 1 or expert_rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    try:
        result = expert_service.submit_expert_review(
            db=db,
            request_id=request_id,
            reviewed_items=reviewed_items,
            expert_feedback=expert_feedback,
            actual_hours=actual_hours,
            expert_rating=expert_rating
        )
        
        return {
            "status": "success",
            "submission": result
        }
        
    except Exception as e:
        logger.error(f"Failed to submit expert review: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/requests/{project_id}")
async def get_project_expert_requests(
    project_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get expert review requests for a project"""
    
    query = db.query(ExpertReviewRequest).filter(
        ExpertReviewRequest.project_id == project_id
    )
    
    if status:
        query = query.filter(ExpertReviewRequest.status == status)
    
    requests = query.order_by(ExpertReviewRequest.created_at.desc()).all()
    
    return {
        "status": "success",
        "project_id": project_id,
        "filter_status": status,
        "total_requests": len(requests),
        "requests": [
            {
                "id": req.id,
                "title": req.title,
                "status": req.status,
                "urgency": req.urgency,
                "items_count": len(req.items_to_review),
                "deadline": req.deadline.isoformat() if req.deadline else None,
                "total_cost": req.total_cost,
                "created_at": req.created_at.isoformat()
            }
            for req in requests
        ]
    } 