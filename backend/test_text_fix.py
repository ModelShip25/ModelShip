#!/usr/bin/env python3
"""
Test script to diagnose text classification issues
"""

import sys
import traceback
import asyncio

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Text Classification - Import Phase")
    print("=" * 50)
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except Exception as e:
        print(f"❌ transformers: {e}")
        return False
        
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
    except Exception as e:
        print(f"❌ torch: {e}")
        return False
        
    try:
        from transformers import pipeline
        print("✅ transformers.pipeline imported")
    except Exception as e:
        print(f"❌ transformers.pipeline: {e}")
        return False
        
    return True

def test_basic_pipeline():
    """Test basic sentiment pipeline"""
    print("\n🧪 Testing Basic Pipeline")
    print("=" * 30)
    
    try:
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis")
        result = classifier("This is a test text")
        print(f"✅ Basic pipeline works: {result}")
        return True
    except Exception as e:
        print(f"❌ Basic pipeline failed: {e}")
        traceback.print_exc()
        return False

def test_text_ml_service():
    """Test our text ML service"""
    print("\n🧪 Testing Text ML Service")
    print("=" * 30)
    
    try:
        from text_ml_service import text_ml_service
        print("✅ text_ml_service imported")
        
        # Test async function
        result = asyncio.run(text_ml_service.classify_text_single(
            "This is a positive test message", 
            "sentiment"
        ))
        print(f"✅ Text classification works: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Text ML service failed: {e}")
        traceback.print_exc()
        return False

def test_text_classification_endpoint():
    """Test text classification router"""
    print("\n🧪 Testing Text Classification Router")
    print("=" * 35)
    
    try:
        from text_classification import router
        print("✅ text_classification router imported")
        return True
    except Exception as e:
        print(f"❌ Text classification router failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 ModelShip Text Classification Diagnostics")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic pipeline
    if not test_basic_pipeline():
        success = False
    
    # Test our service
    if not test_text_ml_service():
        success = False
        
    # Test router
    if not test_text_classification_endpoint():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All text classification tests passed!")
        print("✅ Text annotation should be working")
    else:
        print("❌ Text classification has issues")
        print("🔧 Need to fix the problems above") 