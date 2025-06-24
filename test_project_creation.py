#!/usr/bin/env python3
"""
Test script to check project creation functionality
"""

import requests
import json

def test_simple_endpoint():
    """Test the simple test endpoint"""
    
    url = "http://localhost:8000/api/projects/test"
    
    try:
        print("Testing simple endpoint...")
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Simple endpoint working!")
            return True
        else:
            print("❌ Simple endpoint failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing simple endpoint: {e}")
        return False

def test_simple_create():
    """Test the simple create endpoint"""
    
    url = "http://localhost:8000/api/projects/test-create"
    
    try:
        print("\nTesting simple create endpoint...")
        response = requests.post(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Simple create endpoint working!")
            return True
        else:
            print("❌ Simple create endpoint failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing simple create: {e}")
        return False

def test_project_creation():
    """Test creating a project without authentication"""
    
    url = "http://localhost:8000/api/projects/"
    
    project_data = {
        "name": "Test Project",
        "description": "Testing project creation",
        "project_type": "image_classification",
        "confidence_threshold": 0.8,
        "auto_approve_threshold": 0.95
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("\nTesting project creation...")
        print(f"URL: {url}")
        print(f"Data: {json.dumps(project_data, indent=2)}")
        
        response = requests.post(url, json=project_data, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Project creation successful!")
            return response.json()
        else:
            print("❌ Project creation failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error testing project creation: {e}")
        return None

def test_get_projects():
    """Test getting projects list"""
    
    url = "http://localhost:8000/api/projects/"
    
    try:
        print("\nTesting get projects...")
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Get projects successful!")
            return response.json()
        else:
            print("❌ Get projects failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error testing get projects: {e}")
        return None

def test_server_health():
    """Test if server is running"""
    
    url = "http://localhost:8000/"
    
    try:
        print("Testing server health...")
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
        else:
            print("❌ Server not responding properly!")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("ModelShip Project Creation Test")
    print("=" * 50)
    
    # Test server health first
    if not test_server_health():
        print("Server is not running. Please start the backend server first.")
        exit(1)
    
    # Test simple endpoints first
    test_simple_endpoint()
    test_simple_create()
    
    # Test project creation
    project = test_project_creation()
    
    # Test getting projects
    projects = test_get_projects()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50) 