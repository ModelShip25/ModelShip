"""
Quick test script for advanced ML service
"""

import asyncio
import os
from PIL import Image
import sys
sys.path.append('.')

from advanced_ml_service import advanced_ml_service

async def test_advanced_ml_service():
    """Test the advanced ML service functionality"""
    
    print("Testing Advanced ML Service...")
    print("=" * 50)
    
    # Create a test image
    test_image_path = "test_image.jpg"
    test_image = Image.new('RGB', (224, 224), color='red')
    test_image.save(test_image_path)
    
    try:
        # Test single classification
        print("\n1. Testing single image classification...")
        result = await advanced_ml_service.classify_image_single(
            image_path=test_image_path,
            model_name="resnet50",
            include_metadata=True
        )
        
        print(f"Result: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        print(f"Status: {result['status']}")
        
        # Test batch classification
        print("\n2. Testing batch classification...")
        batch_results = await advanced_ml_service.classify_image_batch(
            image_paths=[test_image_path, test_image_path],
            model_name="resnet50",
            batch_size=2
        )
        
        print(f"Batch results: {len(batch_results)} items processed")
        for i, result in enumerate(batch_results):
            print(f"  Item {i+1}: {result['predicted_label']} ({result['confidence']:.2f})")
        
        # Test performance stats
        print("\n3. Performance statistics:")
        stats = advanced_ml_service.get_performance_stats()
        print(f"Total classifications: {stats.get('total_classifications', 0)}")
        print(f"Average processing time: {stats.get('average_processing_time', 0):.3f}s")
        print(f"Success rate: {stats.get('success_rate', 0):.1f}%")
        
        # Test available models
        print("\n4. Available models:")
        models = advanced_ml_service.get_available_models()
        for model_name, model_info in models.items():
            print(f"  {model_name}: {model_info.get('name', 'Unknown')} - {model_info.get('status', 'Unknown')}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        
    finally:
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

if __name__ == "__main__":
    asyncio.run(test_advanced_ml_service()) 