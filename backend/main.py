from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from file_handler import router as file_router
from classification import router as classify_router
from text_classification import router as text_classify_router
from export import router as export_router
# from billing import router as billing_router  # Temporarily disabled - focusing on core functionality
from review_system import router as review_router
try:
    from advanced_export import router as ml_export_router
except ImportError:
    ml_export_router = None
import uvicorn
import logging
import warnings

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ml_service").setLevel(logging.WARNING)
logging.getLogger("advanced_ml_service").setLevel(logging.WARNING)

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow processor.*")

# Create FastAPI app with metadata
app = FastAPI(
    title="ModelShip API",
    description="AI-powered auto-labeling platform for images and text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth_router)
app.include_router(file_router)
app.include_router(classify_router)
app.include_router(text_classify_router)
app.include_router(export_router)
# app.include_router(billing_router)  # Temporarily disabled - focusing on core functionality
app.include_router(review_router)

# Include advanced ML export router if available
if ml_export_router:
    app.include_router(ml_export_router)

# Add new routers for complete MVP
try:
    from project_management import router as project_router
    app.include_router(project_router)
    logging.info("Project management router loaded")
except ImportError as e:
    logging.warning(f"Project management module not found: {e}")

try:
    from advanced_export_formats import router as advanced_export_router
    app.include_router(advanced_export_router)
    logging.info("Advanced export formats router loaded")
except ImportError as e:
    logging.warning(f"Advanced export formats module not found: {e}")

try:
    from active_learning import router as active_learning_router
    app.include_router(active_learning_router)
    logging.info("Active learning router loaded")
except ImportError as e:
    logging.warning(f"Active learning module not found: {e}")

try:
    from analytics_dashboard import router as analytics_router
    app.include_router(analytics_router)
    logging.info("Analytics dashboard router loaded")
except ImportError as e:
    logging.warning(f"Analytics dashboard module not found: {e}")

try:
    from ml_platform_integration import router as ml_integration_router
    app.include_router(ml_integration_router)
    logging.info("ML platform integration router loaded")
except ImportError as e:
    logging.warning(f"ML platform integration module not found: {e}")

@app.get("/")
def read_root():
    return {
        "message": "ModelShip API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy",
        "features": {
            "auto_labeling": True,
            "image_classification": True,
            "text_classification": True,
            "batch_processing": True,
            "advanced_export": ml_export_router is not None,
            "billing_system": False,  # Temporarily disabled - focusing on core functionality
            "review_system": True,
            "analytics": True
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "ModelShip API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)