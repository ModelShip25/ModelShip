import asyncio
import psutil
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db
from pathlib import Path
import json
import os

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)

class HealthChecker:
    """System health monitoring and checks"""
    
    def __init__(self):
        self.startup_time = datetime.utcnow()
        self.check_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    async def check_database(self, db: Session) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            
            # Simple connection test
            result = db.execute(text("SELECT 1")).fetchone()
            
            # Performance test
            db.execute(text("SELECT COUNT(*) FROM users")).fetchone()
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": round(response_time, 4),
                "message": "Database connection successful"
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed"
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system CPU, memory, and disk usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            
            # Load average (Unix-like systems)
            load_avg = None
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                pass  # Windows doesn't have load average
            
            status = "healthy"
            warnings = []
            
            # Check thresholds
            if cpu_percent > 80:
                status = "warning"
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory_percent > 85:
                status = "warning"
                warnings.append(f"High memory usage: {memory_percent}%")
            
            if disk_percent > 90:
                status = "critical"
                warnings.append(f"High disk usage: {disk_percent}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_mb": round(memory_available / 1024 / 1024, 2),
                "disk_percent": disk_percent,
                "disk_free_gb": round(disk_free / 1024 / 1024 / 1024, 2),
                "load_average": load_avg,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"System resources check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to check system resources"
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space for critical directories"""
        try:
            directories = [
                "storage/uploads",
                "storage/exports", 
                "storage/projects",
                "storage/logs"
            ]
            
            results = {}
            overall_status = "healthy"
            
            for directory in directories:
                if os.path.exists(directory):
                    disk = psutil.disk_usage(directory)
                    free_gb = disk.free / 1024 / 1024 / 1024
                    used_percent = (disk.used / disk.total) * 100
                    
                    status = "healthy"
                    if used_percent > 90:
                        status = "critical"
                        overall_status = "critical"
                    elif used_percent > 80:
                        status = "warning"
                        if overall_status == "healthy":
                            overall_status = "warning"
                    
                    results[directory] = {
                        "status": status,
                        "free_gb": round(free_gb, 2),
                        "used_percent": round(used_percent, 2)
                    }
                else:
                    results[directory] = {
                        "status": "missing",
                        "message": "Directory does not exist"
                    }
            
            return {
                "status": overall_status,
                "directories": results
            }
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_ml_models(self) -> Dict[str, Any]:
        """Check if ML models are loaded and accessible"""
        try:
            models_status = {}
            overall_status = "healthy"
            
            # Check model files
            model_files = [
                "yolov8n.pt",
                "yolov8s.pt"
            ]
            
            for model_file in model_files:
                if os.path.exists(f"backend/{model_file}"):
                    file_size = os.path.getsize(f"backend/{model_file}")
                    models_status[model_file] = {
                        "status": "available",
                        "size_mb": round(file_size / 1024 / 1024, 2)
                    }
                else:
                    models_status[model_file] = {
                        "status": "missing"
                    }
                    overall_status = "warning"
            
            # Check Hugging Face cache
            cache_dir = "models_cache"
            if os.path.exists(cache_dir):
                cache_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(cache_dir)
                    for filename in filenames
                )
                models_status["huggingface_cache"] = {
                    "status": "available",
                    "size_mb": round(cache_size / 1024 / 1024, 2)
                }
            else:
                models_status["huggingface_cache"] = {
                    "status": "missing"
                }
            
            return {
                "status": overall_status,
                "models": models_status
            }
            
        except Exception as e:
            logger.error(f"ML models check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_uptime(self) -> Dict[str, Any]:
        """Get application uptime"""
        uptime = datetime.utcnow() - self.startup_time
        return {
            "startup_time": self.startup_time.isoformat(),
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_human": str(uptime).split('.')[0]  # Remove microseconds
        }
    
    async def comprehensive_health_check(self, db: Session) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()
        
        checks = {
            "database": await self.check_database(db),
            "system_resources": self.check_system_resources(),
            "disk_space": self.check_disk_space(),
            "ml_models": await self.check_ml_models(),
            "uptime": self.get_uptime()
        }
        
        # Determine overall status
        overall_status = "healthy"
        for check_name, check_result in checks.items():
            if check_name == "uptime":
                continue
                
            check_status = check_result.get("status", "unknown")
            if check_status == "critical":
                overall_status = "critical"
                break
            elif check_status in ["unhealthy", "error"]:
                overall_status = "unhealthy"
            elif check_status == "warning" and overall_status == "healthy":
                overall_status = "warning"
        
        result = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "check_duration": round(time.time() - start_time, 4)
        }
        
        # Store in history
        self.check_history.append(result)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)
        
        return result

# Global health checker instance
health_checker = HealthChecker()

@router.get("/")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "ModelShip API is running"
    }

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Comprehensive health check with all subsystems"""
    try:
        result = await health_checker.comprehensive_health_check(db)
        
        # Return appropriate HTTP status code
        if result["status"] == "critical":
            raise HTTPException(status_code=503, detail=result)
        elif result["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=result)
        elif result["status"] == "warning":
            raise HTTPException(status_code=200, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "Health check failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/history")
async def health_check_history():
    """Get health check history"""
    return {
        "history": health_checker.check_history[-10:],  # Last 10 checks
        "total_checks": len(health_checker.check_history)
    }

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        # Current metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / 1024 / 1024 / 1024, 2),
                "memory_total_gb": round(memory.total / 1024 / 1024 / 1024, 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "disk_total_gb": round(disk.total / 1024 / 1024 / 1024, 2)
            },
            "process": {
                "memory_rss_mb": round(process_memory.rss / 1024 / 1024, 2),
                "memory_vms_mb": round(process_memory.vms / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            },
            "uptime": health_checker.get_uptime()
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to collect metrics",
                "message": str(e)
            }
        )

@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Kubernetes readiness probe endpoint"""
    try:
        # Quick database check
        db.execute(text("SELECT 1")).fetchone()
        
        # Check critical directories
        required_dirs = ["storage/uploads", "storage/exports"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                raise HTTPException(
                    status_code=503,
                    detail=f"Required directory missing: {directory}"
                )
        
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe endpoint"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": health_checker.get_uptime()
    } 