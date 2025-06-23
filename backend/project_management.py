from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db
from models import User, Project, LabelSchema, ProjectAssignment, Organization, UserRole, ProjectStatus, LabelType, File as FileModel
from auth import get_current_user
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
import json
import os
import uuid

# Import file storage system
from file_storage import project_storage, user_storage

router = APIRouter(prefix="/api/projects", tags=["project_management"])

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    project_type: str  # 'image_classification', 'text_classification', etc.
    confidence_threshold: float = 0.8
    auto_approve_threshold: float = 0.95
    guidelines: Optional[str] = None
    deadline: Optional[datetime] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    confidence_threshold: Optional[float] = None
    auto_approve_threshold: Optional[float] = None
    guidelines: Optional[str] = None
    deadline: Optional[datetime] = None

class LabelSchemaCreate(BaseModel):
    name: str
    label_type: LabelType
    categories: List[str]
    hierarchy: Optional[Dict[str, Any]] = None
    attributes: Optional[Dict[str, Any]] = None
    is_multi_label: bool = False
    is_hierarchical: bool = False

class ProjectAssignmentCreate(BaseModel):
    user_email: str
    role: UserRole
    assigned_items: int = 0

class ProjectService:
    """Service for managing projects and datasets"""
    
    def __init__(self):
        self.supported_project_types = {
            "image_classification": {
                "description": "Single label per image",
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"],
                "default_schema": ["positive", "negative"]
            },
            "object_detection": {
                "description": "Multiple objects with bounding boxes",
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"],
                "default_schema": ["person", "car", "bike", "animal", "object"]
            },
            "text_classification": {
                "description": "Text document classification",
                "supported_formats": ["txt", "csv", "json"],
                "default_schema": ["positive", "negative", "neutral"]
            },
            "mixed_dataset": {
                "description": "Images and text combined",
                "supported_formats": ["jpg", "jpeg", "png", "txt", "csv"],
                "default_schema": ["category_a", "category_b", "category_c"]
            }
        }

project_service = ProjectService()

@router.post("/")
async def create_project(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new labeling project"""
    
    try:
        # Create project
        project = Project(
            name=project_data.name,
            description=project_data.description,
            project_type=project_data.project_type,
            confidence_threshold=project_data.confidence_threshold,
            auto_approve_threshold=project_data.auto_approve_threshold,
            guidelines=project_data.guidelines,
            deadline=project_data.deadline,
            owner_id=current_user.id,
            organization_id=current_user.organization_id,
            status=ProjectStatus.DRAFT
        )
        
        db.add(project)
        db.commit()
        db.refresh(project)
        
        # Create default assignment for project owner
        owner_assignment = ProjectAssignment(
            project_id=project.id,
            user_id=current_user.id,
            role=UserRole.ADMIN
        )
        
        db.add(owner_assignment)
        db.commit()
        
        # Create default label schema
        default_labels = project_service.supported_project_types[project_data.project_type]["default_schema"]
        
        label_schema = LabelSchema(
            project_id=project.id,
            name="Default Schema",
            label_type=LabelType.OBJECT_DETECTION if project_data.project_type == "object_detection" else LabelType.CLASSIFICATION,
            categories=default_labels,
            is_multi_label=False  # Default for now, can be updated later
        )
        
        db.add(label_schema)
        db.commit()
        
        logger.info(f"Project created: {project.id} by user {current_user.id}")
        
        return {
            "project_id": project.id,
            "name": project.name,
            "status": project.status,
            "project_type": project.project_type,
            "created_at": project.created_at,
            "default_labels": default_labels,
            "message": "Project created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create project: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.post("/create")
async def create_project_alias(
    project_data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new labeling project (alias for POST /)"""
    return await create_project(project_data, current_user, db)

@router.post("/create-test")
async def create_test_project(project_data: ProjectCreate):
    """Create a test project without authentication - for Phase 1 testing"""
    
    try:
        logger.info(f"Creating test project with data: {project_data}")
        
        # Get or create test user
        test_user = user_storage.get_or_create_test_user()
        logger.info(f"Using test user: {test_user['user_id']}")
        
        # Create project using file storage
        project_dict = {
            "name": project_data.name,
            "description": project_data.description,
            "project_type": project_data.project_type,
            "confidence_threshold": project_data.confidence_threshold,
            "auto_approve_threshold": project_data.auto_approve_threshold
        }
        
        project = project_storage.create_project(project_dict, test_user["user_id"])
        logger.info(f"Project created successfully: {project['project_id']}")
        
        return {
            "project_id": project["project_id"],
            "name": project["name"],
            "status": project["status"],
            "project_type": project["project_type"],
            "created_at": project["created_at"],
            "message": "Test project created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create test project: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create test project: {str(e)}")

@router.get("/")
async def get_user_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[ProjectStatus] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0)
):
    """Get all projects accessible to the current user"""
    
    try:
        # Get projects where user is owner or assigned
        query = db.query(Project).join(ProjectAssignment).filter(
            ProjectAssignment.user_id == current_user.id
        )
        
        # Apply status filter
        if status:
            query = query.filter(Project.status == status)
        
        # Apply pagination
        total_count = query.count()
        projects = query.offset(offset).limit(limit).all()
        
        # Format response
        projects_data = []
        for project in projects:
            # Get user's role in this project
            assignment = db.query(ProjectAssignment).filter(
                ProjectAssignment.project_id == project.id,
                ProjectAssignment.user_id == current_user.id
            ).first()
            
            # Calculate progress
            progress_percentage = 0
            if project.total_items > 0:
                progress_percentage = (project.labeled_items / project.total_items) * 100
            
            projects_data.append({
                "project_id": project.id,
                "name": project.name,
                "description": project.description,
                "project_type": project.project_type,
                "status": project.status,
                "user_role": assignment.role if assignment else None,
                "progress": {
                    "total_items": project.total_items,
                    "labeled_items": project.labeled_items,
                    "reviewed_items": project.reviewed_items,
                    "approved_items": project.approved_items,
                    "percentage": round(progress_percentage, 1)
                },
                "created_at": project.created_at,
                "deadline": project.deadline,
                "is_owner": project.owner_id == current_user.id
            })
        
        return {
            "projects": projects_data,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")

@router.get("/test-projects")
async def get_test_projects(
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0)
):
    """Get test projects for Phase 1 testing - no authentication required"""
    
    try:
        # Get test user
        test_user = user_storage.get_or_create_test_user()
        
        # Get projects for test user
        all_projects = project_storage.list_projects(test_user["user_id"])
        
        # Apply pagination
        total_count = len(all_projects)
        projects = all_projects[offset:offset + limit]
        
        return {
            "projects": projects,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Failed to get test projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get test projects: {str(e)}")

@router.get("/{project_id}")
async def get_project_details(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific project"""
    
    try:
        # Check if user has access to this project
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied to this project")
        
        # Get project details
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get label schemas
        label_schemas = db.query(LabelSchema).filter(LabelSchema.project_id == project_id).all()
        
        # Get team assignments
        assignments = db.query(ProjectAssignment).join(User).filter(
            ProjectAssignment.project_id == project_id
        ).all()
        
        team_members = []
        for assignment in assignments:
            team_members.append({
                "user_id": assignment.user.id,
                "email": assignment.user.email,
                "role": assignment.role,
                "assigned_items": assignment.assigned_items,
                "completed_items": assignment.completed_items,
                "completion_rate": (assignment.completed_items / assignment.assigned_items * 100) 
                                 if assignment.assigned_items > 0 else 0
            })
        
        # Calculate detailed progress
        progress_percentage = 0
        if project.total_items > 0:
            progress_percentage = (project.labeled_items / project.total_items) * 100
        
        return {
            "project": {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "project_type": project.project_type,
                "status": project.status,
                "confidence_threshold": project.confidence_threshold,
                "auto_approve_threshold": project.auto_approve_threshold,
                "guidelines": project.guidelines,
                "deadline": project.deadline,
                "created_at": project.created_at,
                "updated_at": project.updated_at
            },
            "progress": {
                "total_items": project.total_items,
                "labeled_items": project.labeled_items,
                "reviewed_items": project.reviewed_items,
                "approved_items": project.approved_items,
                "percentage": round(progress_percentage, 1),
                "automation_rate": round((project.approved_items / project.labeled_items * 100) 
                                       if project.labeled_items > 0 else 0, 1)
            },
            "label_schemas": [
                {
                    "id": schema.id,
                    "name": schema.name,
                    "label_type": schema.label_type,
                    "categories": schema.categories,
                    "is_multi_label": schema.is_multi_label,
                    "is_hierarchical": schema.is_hierarchical
                }
                for schema in label_schemas
            ],
            "team": team_members,
            "user_role": assignment.role
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get project details: {str(e)}")

@router.put("/{project_id}")
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update project information"""
    
    try:
        # Check if user has admin access to this project
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id,
            ProjectAssignment.role.in_([UserRole.ADMIN])
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get and update project
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Update fields
        update_data = project_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(project, field, value)
        
        project.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Project {project_id} updated by user {current_user.id}")
        
        return {
            "project_id": project.id,
            "message": "Project updated successfully",
            "updated_fields": list(update_data.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update project: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@router.post("/{project_id}/label-schema")
async def create_label_schema(
    project_id: int,
    schema_data: LabelSchemaCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a label schema for a project"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id,
            ProjectAssignment.role.in_([UserRole.ADMIN])
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Create label schema
        label_schema = LabelSchema(
            project_id=project_id,
            name=schema_data.name,
            label_type=schema_data.label_type,
            categories=schema_data.categories,
            hierarchy=schema_data.hierarchy,
            attributes=schema_data.attributes,
            is_multi_label=schema_data.is_multi_label,
            is_hierarchical=schema_data.is_hierarchical
        )
        
        db.add(label_schema)
        db.commit()
        db.refresh(label_schema)
        
        logger.info(f"Label schema created for project {project_id}")
        
        return {
            "schema_id": label_schema.id,
            "name": label_schema.name,
            "label_type": label_schema.label_type,
            "categories": label_schema.categories,
            "message": "Label schema created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create label schema: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create label schema: {str(e)}")

@router.post("/{project_id}/assign")
async def assign_user_to_project(
    project_id: int,
    assignment_data: ProjectAssignmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Assign a user to a project with specific role"""
    
    try:
        # Check admin access
        admin_assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id,
            ProjectAssignment.role == UserRole.ADMIN
        ).first()
        
        if not admin_assignment:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Find user by email
        user = db.query(User).filter(User.email == assignment_data.user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user is already assigned
        existing_assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == user.id
        ).first()
        
        if existing_assignment:
            raise HTTPException(status_code=400, detail="User already assigned to this project")
        
        # Create assignment
        assignment = ProjectAssignment(
            project_id=project_id,
            user_id=user.id,
            role=assignment_data.role,
            assigned_items=assignment_data.assigned_items
        )
        
        db.add(assignment)
        db.commit()
        
        logger.info(f"User {user.id} assigned to project {project_id} as {assignment_data.role}")
        
        return {
            "assignment_id": assignment.id,
            "user_email": user.email,
            "role": assignment.role,
            "assigned_items": assignment.assigned_items,
            "message": "User assigned successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assign user: {str(e)}")

@router.get("/{project_id}/analytics")
async def get_project_analytics(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get comprehensive analytics for a project"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Calculate time range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get job and result statistics
        from models import Job, Result
        
        total_jobs = db.query(Job).filter(Job.project_id == project_id).count()
        completed_jobs = db.query(Job).filter(
            Job.project_id == project_id,
            Job.status == "completed"
        ).count()
        
        total_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success"
        ).count()
        
        reviewed_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True
        ).count()
        
        # Calculate accuracy (if we have ground truth data)
        accurate_predictions = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True,
            Result.predicted_label == Result.ground_truth
        ).count()
        
        accuracy = (accurate_predictions / reviewed_results * 100) if reviewed_results > 0 else 0
        
        # Team performance
        team_stats = []
        assignments = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id
        ).all()
        
        for assignment in assignments:
            user = assignment.user
            user_results = db.query(Result).join(Job).filter(
                Job.project_id == project_id,
                Result.reviewed_by == user.id
            ).count()
            
            team_stats.append({
                "user_email": user.email,
                "role": assignment.role,
                "assigned_items": assignment.assigned_items,
                "completed_items": assignment.completed_items,
                "reviewed_items": user_results,
                "completion_rate": (assignment.completed_items / assignment.assigned_items * 100) 
                                 if assignment.assigned_items > 0 else 0
            })
        
        return {
            "project_id": project_id,
            "analytics_period": f"{days} days",
            "overview": {
                "total_items": project.total_items,
                "labeled_items": project.labeled_items,
                "reviewed_items": project.reviewed_items,
                "approved_items": project.approved_items,
                "automation_rate": round((project.approved_items / project.labeled_items * 100) 
                                       if project.labeled_items > 0 else 0, 1),
                "accuracy": round(accuracy, 1)
            },
            "job_statistics": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1)
            },
            "quality_metrics": {
                "total_predictions": total_results,
                "reviewed_predictions": reviewed_results,
                "accurate_predictions": accurate_predictions,
                "review_coverage": round((reviewed_results / total_results * 100) if total_results > 0 else 0, 1)
            },
            "team_performance": team_stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/templates")
async def get_project_templates():
    """Get predefined project templates for common use cases"""
    
    return {
        "templates": {
            "image_classification": {
                "name": "Image Classification Project",
                "description": "Classify images into predefined categories",
                "project_type": "image_classification",
                "recommended_models": ["resnet50", "vit"],
                "confidence_threshold": 0.8,
                "auto_approve_threshold": 0.95,
                "sample_categories": ["cat", "dog", "car", "truck", "person", "building"],
                "use_cases": ["Product categorization", "Content moderation", "Quality control"]
            },
            "sentiment_analysis": {
                "name": "Text Sentiment Analysis",
                "description": "Analyze sentiment in text data",
                "project_type": "text_classification",
                "recommended_models": ["sentiment"],
                "confidence_threshold": 0.7,
                "auto_approve_threshold": 0.9,
                "sample_categories": ["positive", "negative", "neutral"],
                "use_cases": ["Customer feedback", "Social media monitoring", "Review analysis"]
            },
            "content_moderation": {
                "name": "Content Moderation",
                "description": "Detect and filter inappropriate content",
                "project_type": "text_classification",
                "recommended_models": ["toxicity"],
                "confidence_threshold": 0.8,
                "auto_approve_threshold": 0.95,
                "sample_categories": ["safe", "toxic", "spam"],
                "use_cases": ["Social media safety", "Comment filtering", "User-generated content"]
            },
            "document_classification": {
                "name": "Document Classification",
                "description": "Categorize documents by type or topic",
                "project_type": "text_classification",
                "recommended_models": ["topic"],
                "confidence_threshold": 0.75,
                "auto_approve_threshold": 0.9,
                "sample_categories": ["invoice", "contract", "report", "email"],
                "use_cases": ["Document management", "Legal review", "Business automation"]
            }
        },
        "getting_started": {
            "step_1": "Choose a template that matches your use case",
            "step_2": "Create project and customize settings",
            "step_3": "Define or customize label schema",
            "step_4": "Upload your dataset",
            "step_5": "Configure auto-labeling parameters",
            "step_6": "Assign team members and start labeling"
        }
    }

@router.post("/{project_id}/upload")
async def upload_project_files(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload files to a project dataset"""
    
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get supported formats for this project type
    supported_formats = project_service.supported_project_types[project.project_type]["supported_formats"]
    
    # Validate files
    for file in files:
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"File type '.{file_ext}' not supported for {project.project_type}"
            )
    
    # Create upload directory
    upload_dir = os.path.join("uploads", "projects", str(project_id))
    os.makedirs(upload_dir, exist_ok=True)
    
    # Process files
    uploaded_files = []
    for file in files:
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_ext = file.filename.split('.')[-1] if '.' in file.filename else ''
            unique_filename = f"{file_id}.{file_ext}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save file
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Create database record
            file_record = FileModel(
                user_id=current_user.id,
                project_id=project_id,
                filename=unique_filename,
                file_path=file_path,
                file_size=len(contents),
                file_type=file.content_type or f"application/{file_ext}",
                status="uploaded"
            )
            
            db.add(file_record)
            uploaded_files.append({
                "original_filename": file.filename,
                "stored_filename": unique_filename,
                "file_size": len(contents),
                "status": "uploaded"
            })
            
        except Exception as e:
            # Clean up file if database operation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            
            uploaded_files.append({
                "original_filename": file.filename,
                "status": "error",
                "error_message": str(e)
            })
    
    # Update project statistics
    project.total_items += len([f for f in uploaded_files if f["status"] == "uploaded"])
    db.commit()
    
    return {
        "project_id": project_id,
        "uploaded_files": uploaded_files,
        "successful_uploads": len([f for f in uploaded_files if f["status"] == "uploaded"]),
        "failed_uploads": len([f for f in uploaded_files if f["status"] == "error"]),
        "project_total_items": project.total_items
    }

@router.post("/{project_id}/team/invite")
def invite_team_member(
    project_id: int,
    invitation_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Invite a team member to the project"""
    
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Extract invitation details
    user_email = invitation_data.get("email")
    role = invitation_data.get("role", "labeler")
    
    # Validate role
    if role not in [r.value for r in UserRole]:
        raise HTTPException(status_code=400, detail="Invalid role")
    
    # Find user by email
    invited_user = db.query(User).filter(User.email == user_email).first()
    if not invited_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already assigned
    existing_assignment = db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == invited_user.id
    ).first()
    
    if existing_assignment:
        raise HTTPException(status_code=400, detail="User already assigned to this project")
    
    # Create assignment
    assignment = ProjectAssignment(
        project_id=project_id,
        user_id=invited_user.id,
        role=UserRole(role)
    )
    
    db.add(assignment)
    db.commit()
    
    return {
        "message": "Team member invited successfully",
        "project_id": project_id,
        "invited_user": {
            "id": invited_user.id,
            "email": invited_user.email,
            "role": role
        }
    }

@router.put("/{project_id}/settings")
def update_project_settings(
    project_id: int,
    settings_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update project settings"""
    
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Update allowed fields
    if "confidence_threshold" in settings_data:
        threshold = settings_data["confidence_threshold"]
        if 0.0 <= threshold <= 1.0:
            project.confidence_threshold = threshold
    
    if "auto_approve_threshold" in settings_data:
        threshold = settings_data["auto_approve_threshold"]
        if 0.0 <= threshold <= 1.0:
            project.auto_approve_threshold = threshold
    
    if "guidelines" in settings_data:
        project.guidelines = settings_data["guidelines"]
    
    if "status" in settings_data:
        new_status = settings_data["status"]
        if new_status in [s.value for s in ProjectStatus]:
            project.status = ProjectStatus(new_status)
    
    project.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": "Project settings updated successfully",
        "project_id": project_id,
        "updated_settings": {
            "confidence_threshold": project.confidence_threshold,
            "auto_approve_threshold": project.auto_approve_threshold,
            "status": project.status.value,
            "updated_at": project.updated_at
        }
    }

@router.get("/{project_id}/files")
def list_project_files(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    status: Optional[str] = Query(None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0, ge=0)
):
    """List files in a project"""
    
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check access permissions (owner or team member)
    has_access = (project.owner_id == current_user.id) or db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == current_user.id
    ).first()
    
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Build query
    query = db.query(FileModel).filter(FileModel.project_id == project_id)
    
    if status:
        query = query.filter(FileModel.status == status)
    
    files = query.order_by(FileModel.created_at.desc()).offset(offset).limit(limit).all()
    
    # Format response
    files_data = []
    for file in files:
        files_data.append({
            "id": file.id,
            "filename": file.filename,
            "file_size": file.file_size,
            "file_type": file.file_type,
            "status": file.status,
            "created_at": file.created_at,
            "has_results": len(file.results) > 0 if hasattr(file, 'results') else False
        })
    
    return {
        "project_id": project_id,
        "files": files_data,
        "total_count": len(files_data),
        "filters": {"status": status} if status else {}
    }

@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a project (owner only)"""
    
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Archive instead of hard delete
    project.status = ProjectStatus.ARCHIVED
    project.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": "Project archived successfully",
        "project_id": project_id,
        "status": "archived"
    } 