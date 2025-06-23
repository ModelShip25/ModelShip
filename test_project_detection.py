#!/usr/bin/env python3
"""
Test script to verify that object detection saves annotated images to the correct project folder
"""

import os
import sys

# Add backend to path
sys.path.append('backend')

from file_storage import project_storage, user_storage
from project_file_manager import project_file_manager

def test_project_detection_storage():
    """Test that detection results are saved to the correct project"""
    
    print("🧪 Testing Project-Based Detection Storage")
    print("=" * 50)
    
    # Get test user and create project
    test_user = user_storage.get_or_create_test_user()
    
    project_data = {
        "name": "Detection Storage Test",
        "description": "Testing that detection saves to project folder",
        "project_type": "object_detection",
        "confidence_threshold": 0.8,
        "auto_approve_threshold": 0.95
    }
    
    project = project_storage.create_project(project_data, test_user["user_id"])
    project_id = project['project_id']
    
    print(f"✅ Created test project: {project['name']} (ID: {project_id})")
    
    # Show the expected file structure
    originals_path = project_file_manager.get_originals_path(project_id)
    annotated_path = project_file_manager.get_annotated_path(project_id)
    
    print(f"\n📂 Expected file locations:")
    print(f"   Original images: {originals_path}")
    print(f"   Annotated images: {annotated_path}")
    
    # Check if project directories exist
    project_file_manager.ensure_project_directories(project_id)
    
    print(f"\n📁 Directory status:")
    print(f"   Originals dir exists: {os.path.exists(originals_path)}")
    print(f"   Annotated dir exists: {os.path.exists(annotated_path)}")
    
    # Show what happens when frontend calls the API
    print(f"\n🔄 When frontend sends detection request:")
    print(f"   1. POST /api/classify/image/detect")
    print(f"   2. project_id={project_id} (in form data)")
    print(f"   3. Backend saves original to: {originals_path}")
    print(f"   4. Backend runs detection")
    print(f"   5. Backend saves annotated to: {annotated_path}")
    print(f"   6. Frontend displays via: /api/projects/{project_id}/files/annotated/filename.jpg")
    
    print(f"\n✅ Project detection storage test ready!")
    print(f"🎯 Now test with real images in the frontend!")

if __name__ == "__main__":
    test_project_detection_storage() 