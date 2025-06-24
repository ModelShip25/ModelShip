from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from database_base import Base
import os

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./modelship.db"

# Create SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables - import models here to avoid circular imports
def create_tables():
    # Import all models here to register them with Base
    from models import User, File, Job, Result, Project
    
    # Import Phase 3 models
    try:
        from vertical_templates import VerticalTemplate
        from expert_in_loop import Expert, ExpertReviewRequest
        from bias_fairness_reports import BiasReport
        from security_compliance import SecurityAuditLog, ComplianceReport, EncryptionKey
        from ml_assisted_prelabeling import PreLabelingModel, PreLabelingResult
        from consensus_controls import ConsensusTask, AnnotatorAssignment, AnnotatorProfile
    except ImportError as e:
        print(f"Warning: Could not import some models: {e}")
    
    Base.metadata.create_all(bind=engine)

# Initialize database on import
if __name__ != "__main__":
    create_tables()