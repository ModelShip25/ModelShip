#!/usr/bin/env python3
"""Test the simple project API"""

import requests
import json
import time
import threading
import sys

def test_project_api():
    """Test the simple project API"""
    
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Simple Project API...")
    
    try:
        # Test the test-create endpoint first
        print("1. Testing quick project creation...")
        response = requests.post(f"{base_url}/api/simple-projects/test-create", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Test project created: {data}")
            return True
        else:
            print(f"❌ Test creation failed: {response.status_code} - {response.text}")
            
        # Test manual project creation
        print("2. Testing manual project creation...")
        project_data = {
            "name": "Manual Test Project",
            "project_type": "image_classification",
            "description": "Testing manual creation"
        }
        
        response = requests.post(f"{base_url}/api/simple-projects/", json=project_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Manual project created: {data}")
            return True
        else:
            print(f"❌ Manual creation failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure server is running on port 8000")
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    return False

if __name__ == "__main__":
    # Simple check if server is accessible
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        if response.status_code == 200:
            test_project_api()
        else:
            print("❌ Server not responding correctly")
    except:
        print("❌ Server not running. Start with: python main.py") 