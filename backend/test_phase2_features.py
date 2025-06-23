"""
Phase 2 Feature Testing Script
Tests all advanced annotation features: Quality Dashboard, MLOps, Versioning, Gold Standard Testing
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class Phase2Tester:
    def __init__(self):
        self.base_url = BASE_URL
        self.project_id = 1  # Assuming project 1 exists
        self.results = {}
    
    def test_quality_dashboard(self) -> Dict[str, Any]:
        """Test quality dashboard endpoints"""
        print("\n🔍 Testing Quality Dashboard...")
        
        results = {}
        
        # Test quality metrics
        try:
            response = requests.get(f"{self.base_url}/api/quality/metrics/{self.project_id}")
            results["metrics"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Quality Metrics: {response.status_code}")
        except Exception as e:
            results["metrics"] = {"success": False, "error": str(e)}
            print(f"   ❌ Quality Metrics failed: {e}")
        
        # Test annotator performance
        try:
            response = requests.get(f"{self.base_url}/api/quality/annotators/{self.project_id}")
            results["annotators"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Annotator Performance: {response.status_code}")
        except Exception as e:
            results["annotators"] = {"success": False, "error": str(e)}
            print(f"   ❌ Annotator Performance failed: {e}")
        
        # Test trends
        try:
            response = requests.get(f"{self.base_url}/api/quality/trends/{self.project_id}")
            results["trends"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Annotation Trends: {response.status_code}")
        except Exception as e:
            results["trends"] = {"success": False, "error": str(e)}
            print(f"   ❌ Annotation Trends failed: {e}")
        
        # Test alerts
        try:
            response = requests.get(f"{self.base_url}/api/quality/alerts/{self.project_id}")
            results["alerts"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Quality Alerts: {response.status_code}")
        except Exception as e:
            results["alerts"] = {"success": False, "error": str(e)}
            print(f"   ❌ Quality Alerts failed: {e}")
        
        # Test complete dashboard
        try:
            response = requests.get(f"{self.base_url}/api/quality/dashboard/{self.project_id}")
            results["dashboard"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Complete Dashboard: {response.status_code}")
        except Exception as e:
            results["dashboard"] = {"success": False, "error": str(e)}
            print(f"   ❌ Complete Dashboard failed: {e}")
        
        return results
    
    def test_mlops_integration(self) -> Dict[str, Any]:
        """Test MLOps integration endpoints"""
        print("\n🚀 Testing MLOps Integration...")
        
        results = {}
        
        # Test supported platforms
        try:
            response = requests.get(f"{self.base_url}/api/mlops/platforms")
            results["platforms"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Supported Platforms: {response.status_code}")
        except Exception as e:
            results["platforms"] = {"success": False, "error": str(e)}
            print(f"   ❌ Supported Platforms failed: {e}")
        
        # Test export for training
        try:
            export_data = {
                "platform": "mlflow",
                "config": {
                    "experiment_name": "ModelShip_Test",
                    "run_name": f"test_run_{int(time.time())}"
                }
            }
            response = requests.post(
                f"{self.base_url}/api/mlops/export/{self.project_id}",
                params={"platform": "mlflow"},
                json=export_data["config"]
            )
            results["export"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Export for Training: {response.status_code}")
        except Exception as e:
            results["export"] = {"success": False, "error": str(e)}
            print(f"   ❌ Export for Training failed: {e}")
        
        # Test training pipeline trigger
        try:
            training_data = {
                "platform": "mlflow",
                "dataset_info": {"total_samples": 100, "labels": ["cat", "dog"]},
                "training_config": {"epochs": 10, "batch_size": 32}
            }
            response = requests.post(
                f"{self.base_url}/api/mlops/train",
                json=training_data
            )
            results["training"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Training Pipeline: {response.status_code}")
        except Exception as e:
            results["training"] = {"success": False, "error": str(e)}
            print(f"   ❌ Training Pipeline failed: {e}")
        
        return results
    
    def test_data_versioning(self) -> Dict[str, Any]:
        """Test data versioning endpoints"""
        print("\n📦 Testing Data Versioning...")
        
        results = {}
        
        # Test create version
        try:
            version_data = {
                "description": f"Test version created at {datetime.now()}",
                "version_type": "minor",
                "user_id": 1
            }
            response = requests.post(
                f"{self.base_url}/api/versioning/create/{self.project_id}",
                params=version_data
            )
            results["create"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Create Version: {response.status_code}")
            
            # Store version ID for later tests
            if response.status_code == 200:
                self.test_version_id = response.json().get("version", {}).get("id")
        except Exception as e:
            results["create"] = {"success": False, "error": str(e)}
            print(f"   ❌ Create Version failed: {e}")
        
        # Test list versions
        try:
            response = requests.get(f"{self.base_url}/api/versioning/list/{self.project_id}")
            results["list"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ List Versions: {response.status_code}")
        except Exception as e:
            results["list"] = {"success": False, "error": str(e)}
            print(f"   ❌ List Versions failed: {e}")
        
        # Test version diff (if we have a version ID)
        if hasattr(self, 'test_version_id') and self.test_version_id:
            try:
                response = requests.get(f"{self.base_url}/api/versioning/diff/{self.test_version_id}")
                results["diff"] = {
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "data": response.json() if response.status_code == 200 else None
                }
                print(f"   ✅ Version Diff: {response.status_code}")
            except Exception as e:
                results["diff"] = {"success": False, "error": str(e)}
                print(f"   ❌ Version Diff failed: {e}")
        
        return results
    
    def test_gold_standard(self) -> Dict[str, Any]:
        """Test gold standard testing endpoints"""
        print("\n🏆 Testing Gold Standard Testing...")
        
        results = {}
        
        # Test create gold sample
        try:
            sample_data = {
                "filename": "test_gold_sample.jpg",
                "file_path": "/test/path/test_gold_sample.jpg",
                "correct_label": "cat",
                "sample_type": "image_classification",
                "difficulty_level": "medium",
                "description": "Test gold standard sample",
                "user_id": 1
            }
            response = requests.post(
                f"{self.base_url}/api/gold-standard/samples/create/{self.project_id}",
                params=sample_data
            )
            results["create_sample"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Create Gold Sample: {response.status_code}")
        except Exception as e:
            results["create_sample"] = {"success": False, "error": str(e)}
            print(f"   ❌ Create Gold Sample failed: {e}")
        
        # Test list gold samples
        try:
            response = requests.get(f"{self.base_url}/api/gold-standard/samples/{self.project_id}")
            results["list_samples"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ List Gold Samples: {response.status_code}")
        except Exception as e:
            results["list_samples"] = {"success": False, "error": str(e)}
            print(f"   ❌ List Gold Samples failed: {e}")
        
        # Test inject gold samples (requires a job ID, using 1 as test)
        try:
            inject_data = {
                "injection_rate": "medium",
                "total_samples": 50
            }
            response = requests.post(
                f"{self.base_url}/api/gold-standard/inject/{self.project_id}/1",
                params=inject_data
            )
            results["inject"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Inject Gold Samples: {response.status_code}")
        except Exception as e:
            results["inject"] = {"success": False, "error": str(e)}
            print(f"   ❌ Inject Gold Samples failed: {e}")
        
        # Test performance metrics
        try:
            response = requests.get(f"{self.base_url}/api/gold-standard/performance/{self.project_id}")
            results["performance"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Performance Metrics: {response.status_code}")
        except Exception as e:
            results["performance"] = {"success": False, "error": str(e)}
            print(f"   ❌ Performance Metrics failed: {e}")
        
        # Test drift detection
        try:
            response = requests.get(f"{self.base_url}/api/gold-standard/drift-detection/{self.project_id}")
            results["drift"] = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "data": response.json() if response.status_code == 200 else None
            }
            print(f"   ✅ Drift Detection: {response.status_code}")
        except Exception as e:
            results["drift"] = {"success": False, "error": str(e)}
            print(f"   ❌ Drift Detection failed: {e}")
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 2 feature tests"""
        print("🚀 Starting Phase 2 Feature Testing...")
        print(f"Testing against: {self.base_url}")
        print(f"Project ID: {self.project_id}")
        
        # Run all test suites
        self.results["quality_dashboard"] = self.test_quality_dashboard()
        self.results["mlops_integration"] = self.test_mlops_integration()
        self.results["data_versioning"] = self.test_data_versioning()
        self.results["gold_standard"] = self.test_gold_standard()
        
        # Generate summary
        self.generate_test_summary()
        
        return self.results
    
    def generate_test_summary(self):
        """Generate and print test summary"""
        print("\n" + "="*60)
        print("📊 PHASE 2 TESTING SUMMARY")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for suite_name, suite_results in self.results.items():
            print(f"\n{suite_name.upper().replace('_', ' ')}:")
            suite_passed = 0
            suite_total = 0
            
            for test_name, test_result in suite_results.items():
                suite_total += 1
                total_tests += 1
                
                if test_result.get("success", False):
                    suite_passed += 1
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                print(f"  {test_name}: {status}")
                if not test_result.get("success", False) and "error" in test_result:
                    print(f"    Error: {test_result['error']}")
            
            print(f"  Suite Score: {suite_passed}/{suite_total}")
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL PHASE 2 FEATURES WORKING PERFECTLY!")
        elif passed_tests > total_tests * 0.8:
            print("\n✅ Phase 2 features mostly working - minor issues to address")
        else:
            print("\n⚠️  Phase 2 features need attention - several failures detected")

def main():
    """Main testing function"""
    tester = Phase2Tester()
    
    # Check if server is running
    try:
        response = requests.get(f"{tester.base_url}/")
        if response.status_code != 200:
            print(f"❌ Server not responding at {tester.base_url}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the backend server is running with: python main.py")
        return
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Save results to file
    with open("phase2_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: phase2_test_results.json")

if __name__ == "__main__":
    main() 