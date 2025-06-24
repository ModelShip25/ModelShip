from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
from database_base import Base
import enum

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    LABELER = "labeler" 
    REVIEWER = "reviewer"
    VIEWER = "viewer"

class ProjectStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class LabelType(str, enum.Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    TEXT_CLASSIFICATION = "text_classification"
    NAMED_ENTITY = "named_entity"
    CUSTOM = "custom"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    subscription_tier = Column(String(50), default="free")
    credits_remaining = Column(Integer, default=100)
    role = Column(SQLEnum(UserRole), default=UserRole.LABELER)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    files = relationship("File", back_populates="user")
    jobs = relationship("Job", back_populates="user")
    owned_projects = relationship("Project", foreign_keys="Project.owner_id", back_populates="owner")
    project_assignments = relationship("ProjectAssignment", back_populates="user")

class Organization(Base):
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    plan_type = Column(String(50), default="team")
    credits_pool = Column(Integer, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="organization")
    projects = relationship("Project", back_populates="organization")

class Project(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    project_type = Column(String(50), nullable=False)  # 'image_classification', 'text_classification', etc.
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.DRAFT)
    
    # Project configuration
    confidence_threshold = Column(Float, default=0.8)
    auto_approve_threshold = Column(Float, default=0.95)
    guidelines = Column(Text, nullable=True)
    
    # Ownership and organization
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Allow null for guest projects
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True)
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    labeled_items = Column(Integer, default=0)
    reviewed_items = Column(Integer, default=0)
    approved_items = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deadline = Column(DateTime, nullable=True)
    
    # Relationships
    owner = relationship("User", foreign_keys=[owner_id], back_populates="owned_projects")
    organization = relationship("Organization", back_populates="projects")
    label_schemas = relationship("LabelSchema", back_populates="project")
    assignments = relationship("ProjectAssignment", back_populates="project")
    jobs = relationship("Job", back_populates="project")
    files = relationship("File", back_populates="project")

class LabelSchema(Base):
    __tablename__ = "label_schemas"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    label_type = Column(SQLEnum(LabelType), nullable=False)
    
    # Schema definition (JSON structure)
    categories = Column(JSON, nullable=False)  # List of label categories
    hierarchy = Column(JSON, nullable=True)   # Hierarchical structure if needed
    attributes = Column(JSON, nullable=True)  # Additional attributes per label
    
    # Configuration
    is_multi_label = Column(Boolean, default=False)
    is_hierarchical = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="label_schemas")

class ProjectAssignment(Base):
    __tablename__ = "project_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(SQLEnum(UserRole), nullable=False)
    
    # Assignment details
    assigned_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="assignments")
    user = relationship("User", back_populates="project_assignments")

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Allow null for guest uploads
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(50))
    status = Column(String(50), default="uploaded")
    
    # File metadata

    file_metadata = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="files")
    project = relationship("Project", back_populates="files")
    results = relationship("Result", back_populates="file")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Allow null for guest jobs
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    job_type = Column(String(50), nullable=False)  # 'image' or 'text'
    status = Column(String(50), default="processing")  # 'queued', 'processing', 'completed', 'failed'
    
    # Job configuration
    model_name = Column(String(100), nullable=True)
    confidence_threshold = Column(Float, default=0.8)
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    completed_items = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    project = relationship("Project", back_populates="jobs")
    results = relationship("Result", back_populates="job")

class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    filename = Column(String(255))
    
    # Prediction results
    predicted_label = Column(String(255))
    confidence = Column(Float)
    all_predictions = Column(JSON, nullable=True)  # Store top-k predictions
    
    # Processing metadata
    processing_time = Column(Float)
    model_version = Column(String(100), nullable=True)
    
    # Review and quality control
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_action = Column(String(50), nullable=True)  # 'approved', 'corrected', 'rejected'
    
    # Ground truth and corrections
    ground_truth = Column(String(255))
    correction_reason = Column(Text, nullable=True)
    
    # Status and errors
    status = Column(String(50), default="success")  # 'success', 'error', 'pending_review'
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="results")
    file = relationship("File", back_populates="results")
    reviewer = relationship("User", foreign_keys=[reviewed_by])

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Review details
    action = Column(String(50), nullable=False)  # 'approve', 'reject', 'modify'
    original_label = Column(String(255))
    corrected_label = Column(String(255), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Review metadata
    review_time_seconds = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    result = relationship("Result")
    reviewer = relationship("User", foreign_keys=[reviewer_id])

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Metric data
    metric_type = Column(String(100), nullable=False)  # 'accuracy', 'throughput', 'cost_savings', etc.
    metric_value = Column(Float, nullable=False)
    metric_data = Column(JSON, nullable=True)  # Additional metric details
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow) 