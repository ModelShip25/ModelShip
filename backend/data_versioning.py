from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from database import get_db, Base
from models import Project, Job, Result
from typing import Dict, List, Any, Optional
import logging
import json
import hashlib
from datetime import datetime
import copy

logger = logging.getLogger(__name__)

# Data versioning models
class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    version_number = Column(String(50), nullable=False)
    version_hash = Column(String(64), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=False)
    
    # Snapshot metadata
    total_annotations = Column(Integer, default=0)
    annotation_summary = Column(JSON)  # Label counts, confidence stats, etc.
    
    # Version relationships
    parent_version_id = Column(Integer, ForeignKey("dataset_versions.id"))
    
class AnnotationSnapshot(Base):
    __tablename__ = "annotation_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    
    # Snapshot of the annotation at this version
    filename = Column(String(255), nullable=False)
    predicted_label = Column(String(255))
    ground_truth = Column(String(255))
    confidence = Column(Float)
    bounding_boxes = Column(JSON)
    entities = Column(JSON)
    review_action = Column(String(50))
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    
    # Change tracking
    change_type = Column(String(50))  # created, modified, deleted
    previous_value = Column(JSON)  # Previous state for rollback
    
class DataVersioningService:
    def __init__(self):
        self.version_cache = {}
    
    def create_version(
        self,
        db: Session,
        project_id: int,
        description: str,
        user_id: int,
        version_type: str = "manual"
    ) -> Dict[str, Any]:
        """Create a new version of the dataset"""
        
        # Get current project state
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Get all current results
        current_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id
        ).all()
        
        if not current_results:
            raise ValueError("No annotations found to version")
        
        # Calculate version hash based on current state
        version_hash = self._calculate_version_hash(current_results)
        
        # Check if this exact version already exists
        existing_version = db.query(DatasetVersion).filter(
            DatasetVersion.version_hash == version_hash
        ).first()
        
        if existing_version:
            return {
                "status": "exists",
                "version": self._format_version_info(existing_version),
                "message": "This exact version already exists"
            }
        
        # Generate version number
        latest_version = db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id
        ).order_by(DatasetVersion.id.desc()).first()
        
        if latest_version:
            version_parts = latest_version.version_number.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            if version_type == "major":
                version_number = f"{major + 1}.0"
            else:
                version_number = f"{major}.{minor + 1}"
        else:
            version_number = "1.0"
        
        # Create annotation summary
        annotation_summary = self._create_annotation_summary(current_results)
        
        # Create version record
        new_version = DatasetVersion(
            project_id=project_id,
            version_number=version_number,
            version_hash=version_hash,
            description=description,
            created_by=user_id,
            total_annotations=len(current_results),
            annotation_summary=annotation_summary,
            is_active=True
        )
        
        # Deactivate previous active version
        db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id,
            DatasetVersion.is_active == True
        ).update({"is_active": False})
        
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        
        # Create snapshots of all annotations
        self._create_annotation_snapshots(db, new_version.id, current_results)
        
        logger.info(f"Created version {version_number} for project {project_id}")
        
        return {
            "status": "created",
            "version": self._format_version_info(new_version),
            "snapshots_created": len(current_results)
        }
    
    def _calculate_version_hash(self, results: List[Result]) -> str:
        """Calculate hash representing the current state of annotations"""
        
        # Sort results by ID for consistent hashing
        sorted_results = sorted(results, key=lambda r: r.id)
        
        hash_data = []
        for result in sorted_results:
            result_data = {
                "id": result.id,
                "filename": result.filename,
                "predicted_label": result.predicted_label,
                "ground_truth": result.ground_truth,
                "confidence": result.confidence,
                "reviewed": result.reviewed,
                "review_action": result.review_action
            }
            hash_data.append(json.dumps(result_data, sort_keys=True))
        
        combined_data = "|".join(hash_data)
        return hashlib.sha256(combined_data.encode()).hexdigest()
    
    def _create_annotation_summary(self, results: List[Result]) -> Dict[str, Any]:
        """Create summary statistics for the annotation set"""
        
        label_counts = {}
        confidence_scores = []
        review_actions = {"approve": 0, "reject": 0, "modify": 0, "none": 0}
        
        for result in results:
            # Count labels
            label = result.ground_truth or result.predicted_label
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Collect confidence scores
            if result.confidence:
                confidence_scores.append(result.confidence)
            
            # Count review actions
            action = result.review_action or "none"
            review_actions[action] = review_actions.get(action, 0) + 1
        
        return {
            "total_annotations": len(results),
            "unique_labels": len(label_counts),
            "label_distribution": label_counts,
            "confidence_stats": {
                "mean": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0
            },
            "review_distribution": review_actions,
            "reviewed_percentage": (sum(1 for r in results if r.reviewed) / len(results)) * 100
        }
    
    def _create_annotation_snapshots(self, db: Session, version_id: int, results: List[Result]):
        """Create snapshots of all annotations for this version"""
        
        snapshots = []
        for result in results:
            snapshot = AnnotationSnapshot(
                version_id=version_id,
                result_id=result.id,
                filename=result.filename,
                predicted_label=result.predicted_label,
                ground_truth=result.ground_truth,
                confidence=result.confidence,
                bounding_boxes=result.bounding_boxes,
                entities=result.entities,
                review_action=result.review_action,
                reviewed_by=result.reviewed_by,
                change_type="snapshot"
            )
            snapshots.append(snapshot)
        
        db.add_all(snapshots)
        db.commit()
    
    def get_versions(self, db: Session, project_id: int) -> List[Dict[str, Any]]:
        """Get all versions for a project"""
        
        versions = db.query(DatasetVersion).filter(
            DatasetVersion.project_id == project_id
        ).order_by(DatasetVersion.created_at.desc()).all()
        
        return [self._format_version_info(version) for version in versions]
    
    def _format_version_info(self, version: DatasetVersion) -> Dict[str, Any]:
        """Format version information for API response"""
        
        return {
            "id": version.id,
            "version_number": version.version_number,
            "version_hash": version.version_hash[:12],  # Short hash for display
            "description": version.description,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by,
            "is_active": version.is_active,
            "total_annotations": version.total_annotations,
            "annotation_summary": version.annotation_summary
        }
    
    def compare_versions(
        self,
        db: Session,
        version1_id: int,
        version2_id: int
    ) -> Dict[str, Any]:
        """Compare two versions and show differences"""
        
        version1 = db.query(DatasetVersion).filter(DatasetVersion.id == version1_id).first()
        version2 = db.query(DatasetVersion).filter(DatasetVersion.id == version2_id).first()
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Get snapshots for both versions
        snapshots1 = db.query(AnnotationSnapshot).filter(
            AnnotationSnapshot.version_id == version1_id
        ).all()
        
        snapshots2 = db.query(AnnotationSnapshot).filter(
            AnnotationSnapshot.version_id == version2_id
        ).all()
        
        # Create lookup dictionaries
        snap1_dict = {snap.filename: snap for snap in snapshots1}
        snap2_dict = {snap.filename: snap for snap in snapshots2}
        
        # Find differences
        added_files = set(snap2_dict.keys()) - set(snap1_dict.keys())
        removed_files = set(snap1_dict.keys()) - set(snap2_dict.keys())
        common_files = set(snap1_dict.keys()) & set(snap2_dict.keys())
        
        modified_files = []
        for filename in common_files:
            snap1 = snap1_dict[filename]
            snap2 = snap2_dict[filename]
            
            if (snap1.ground_truth != snap2.ground_truth or 
                snap1.predicted_label != snap2.predicted_label or
                snap1.review_action != snap2.review_action):
                
                modified_files.append({
                    "filename": filename,
                    "changes": {
                        "ground_truth": {
                            "old": snap1.ground_truth,
                            "new": snap2.ground_truth
                        },
                        "predicted_label": {
                            "old": snap1.predicted_label,
                            "new": snap2.predicted_label
                        },
                        "review_action": {
                            "old": snap1.review_action,
                            "new": snap2.review_action
                        }
                    }
                })
        
        return {
            "version1": self._format_version_info(version1),
            "version2": self._format_version_info(version2),
            "comparison": {
                "added_annotations": len(added_files),
                "removed_annotations": len(removed_files),
                "modified_annotations": len(modified_files),
                "unchanged_annotations": len(common_files) - len(modified_files),
                "added_files": list(added_files)[:10],  # Limit for response size
                "removed_files": list(removed_files)[:10],
                "modified_files": modified_files[:10]
            }
        }
    
    def rollback_to_version(
        self,
        db: Session,
        project_id: int,
        target_version_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """Rollback project to a specific version"""
        
        target_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == target_version_id,
            DatasetVersion.project_id == project_id
        ).first()
        
        if not target_version:
            raise ValueError("Target version not found")
        
        # Get snapshots for target version
        target_snapshots = db.query(AnnotationSnapshot).filter(
            AnnotationSnapshot.version_id == target_version_id
        ).all()
        
        # Create rollback plan
        rollback_actions = []
        
        for snapshot in target_snapshots:
            # Find current result
            current_result = db.query(Result).filter(
                Result.id == snapshot.result_id
            ).first()
            
            if current_result:
                # Store current state for potential undo
                current_state = {
                    "predicted_label": current_result.predicted_label,
                    "ground_truth": current_result.ground_truth,
                    "confidence": current_result.confidence,
                    "review_action": current_result.review_action,
                    "reviewed_by": current_result.reviewed_by
                }
                
                # Apply rollback
                current_result.predicted_label = snapshot.predicted_label
                current_result.ground_truth = snapshot.ground_truth
                current_result.confidence = snapshot.confidence
                current_result.review_action = snapshot.review_action
                current_result.reviewed_by = snapshot.reviewed_by
                
                rollback_actions.append({
                    "result_id": current_result.id,
                    "filename": snapshot.filename,
                    "previous_state": current_state,
                    "rolled_back_to": {
                        "predicted_label": snapshot.predicted_label,
                        "ground_truth": snapshot.ground_truth,
                        "review_action": snapshot.review_action
                    }
                })
        
        # Create new version for the rollback
        rollback_description = f"Rollback to version {target_version.version_number}"
        rollback_version = self.create_version(
            db, project_id, rollback_description, user_id, "rollback"
        )
        
        db.commit()
        
        logger.info(f"Rolled back project {project_id} to version {target_version.version_number}")
        
        return {
            "status": "success",
            "rolled_back_to": self._format_version_info(target_version),
            "new_version": rollback_version["version"],
            "actions_performed": len(rollback_actions),
            "rollback_summary": rollback_actions[:5]  # First 5 for response
        }
    
    def get_version_diff(
        self,
        db: Session,
        version_id: int,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed diff for a specific version or file"""
        
        version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if not version:
            raise ValueError("Version not found")
        
        query = db.query(AnnotationSnapshot).filter(
            AnnotationSnapshot.version_id == version_id
        )
        
        if filename:
            query = query.filter(AnnotationSnapshot.filename == filename)
        
        snapshots = query.all()
        
        diff_data = []
        for snapshot in snapshots:
            # Get current state
            current_result = db.query(Result).filter(
                Result.id == snapshot.result_id
            ).first()
            
            if current_result:
                diff_data.append({
                    "filename": snapshot.filename,
                    "snapshot_state": {
                        "predicted_label": snapshot.predicted_label,
                        "ground_truth": snapshot.ground_truth,
                        "confidence": snapshot.confidence,
                        "review_action": snapshot.review_action
                    },
                    "current_state": {
                        "predicted_label": current_result.predicted_label,
                        "ground_truth": current_result.ground_truth,
                        "confidence": current_result.confidence,
                        "review_action": current_result.review_action
                    },
                    "has_changes": (
                        snapshot.predicted_label != current_result.predicted_label or
                        snapshot.ground_truth != current_result.ground_truth or
                        snapshot.review_action != current_result.review_action
                    )
                })
        
        return {
            "version": self._format_version_info(version),
            "filename_filter": filename,
            "total_items": len(diff_data),
            "changed_items": len([d for d in diff_data if d["has_changes"]]),
            "diff_data": diff_data
        }

# Create service instance
versioning_service = DataVersioningService()

# FastAPI Router
router = APIRouter(prefix="/api/versioning", tags=["versioning"])

@router.post("/create/{project_id}")
async def create_version(
    project_id: int,
    description: str,
    version_type: str = "minor",
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Create a new version of the dataset"""
    try:
        result = versioning_service.create_version(
            db, project_id, description, user_id, version_type
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create version: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list/{project_id}")
async def list_versions(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get all versions for a project"""
    try:
        versions = versioning_service.get_versions(db, project_id)
        return {
            "status": "success",
            "project_id": project_id,
            "versions": versions,
            "total_versions": len(versions)
        }
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/compare/{version1_id}/{version2_id}")
async def compare_versions(
    version1_id: int,
    version2_id: int,
    db: Session = Depends(get_db)
):
    """Compare two versions"""
    try:
        comparison = versioning_service.compare_versions(db, version1_id, version2_id)
        return {
            "status": "success",
            "comparison": comparison
        }
    except Exception as e:
        logger.error(f"Failed to compare versions: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/rollback/{project_id}/{target_version_id}")
async def rollback_to_version(
    project_id: int,
    target_version_id: int,
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Rollback project to a specific version"""
    try:
        result = versioning_service.rollback_to_version(
            db, project_id, target_version_id, user_id
        )
        return result
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/diff/{version_id}")
async def get_version_diff(
    version_id: int,
    filename: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get detailed diff for a version"""
    try:
        diff = versioning_service.get_version_diff(db, version_id, filename)
        return {
            "status": "success",
            "diff": diff
        }
    except Exception as e:
        logger.error(f"Failed to get diff: {e}")
        raise HTTPException(status_code=400, detail=str(e)) 