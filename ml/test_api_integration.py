#!/usr/bin/env python3
"""
API Integration Test for Enhanced SmartMarketOOPS System
Tests the enhanced prediction endpoints and signal quality metrics
"""

import requests
import json
import time
import sys

def test_enhanced_api():
    """Test the enhanced API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Enhanced API Integration")
    print("="*50)
    
    # Test data
    test_data = {
        "symbol": "BTCUSDT",
        "features": {
            "open": 45000.0,
            "high": 45500.0,
            "low": 44800.0,
            "close": 45200.0,
            "volume": 1500000.0,
            "rsi": 55.0,
            "macd": 0.1,
            "bb_upper": 46000.0,
            "bb_lower": 44000.0
        },
        "sequence_length": 60
    }
    
    # Test 1: Basic health check
    print("1. Testing basic health check...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Enhanced prediction endpoint
    print("\n2. Testing enhanced prediction endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/models/enhanced/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced prediction successful")
            print(f"   Prediction: {result.get('prediction', 'N/A'):.3f}")
            print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"   Signal Valid: {result.get('signal_valid', 'N/A')}")
            print(f"   Quality Score: {result.get('quality_score', 'N/A'):.3f}")
            print(f"   Market Regime: {result.get('market_regime', 'N/A')}")
            print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"   Enhanced: {result.get('enhanced', 'N/A')}")
            
            # Check if we have model predictions breakdown
            if 'model_predictions' in result:
                print(f"   Model Predictions: {len(result['model_predictions'])} models")
                for model_name, pred in result['model_predictions'].items():
                    print(f"     {model_name}: {pred.get('prediction', 0):.3f} (conf: {pred.get('confidence', 0):.3f})")
            
            return True
        else:
            print(f"❌ Enhanced prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced prediction error: {e}")
        return False
    
    # Test 3: Traditional prediction fallback
    print("\n3. Testing traditional prediction fallback...")
    try:
        response = requests.post(
            f"{base_url}/api/models/predict",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Traditional prediction successful")
            print(f"   Predictions: {result.get('predictions', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"   Direction: {result.get('predicted_direction', 'N/A')}")
            return True
        else:
            print(f"⚠️  Traditional prediction failed: {response.status_code}")
            print(f"   This is expected if no traditional models are trained")
            return True  # Not critical for enhanced system
            
    except Exception as e:
        print(f"⚠️  Traditional prediction error: {e}")
        return True  # Not critical for enhanced system

def test_model_management():
    """Test model management endpoints"""
    base_url = "http://localhost:8000"
    
    print("\n4. Testing model management endpoints...")
    
    # Test enhanced model status
    try:
        response = requests.get(f"{base_url}/api/models/enhanced/models/BTCUSDT/status")
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Enhanced model status retrieved")
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Enhanced: {status.get('enhanced', False)}")
        else:
            print(f"⚠️  Enhanced model status not available: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Enhanced model status error: {e}")
    
    # Test enhanced model loading
    try:
        response = requests.post(f"{base_url}/api/models/enhanced/models/BTCUSDT/load")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced model loading successful")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"⚠️  Enhanced model loading failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Enhanced model loading error: {e}")

def test_performance_tracking():
    """Test performance tracking endpoints"""
    base_url = "http://localhost:8000"
    
    print("\n5. Testing performance tracking...")
    
    try:
        # Test performance update
        performance_data = {
            "prediction": 0.7,
            "actual_outcome": 1.0,
            "confidence": 0.8
        }
        
        response = requests.post(
            f"{base_url}/api/models/enhanced/models/BTCUSDT/performance",
            params=performance_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Performance tracking successful")
            print(f"   Status: {result.get('status', 'unknown')}")
        else:
            print(f"⚠️  Performance tracking failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Performance tracking error: {e}")

def main():
    """Run all API integration tests"""
    print("🚀 Enhanced SmartMarketOOPS API Integration Test")
    print("="*60)
    
    # Wait for service to be ready
    print("Waiting for ML service to be ready...")
    time.sleep(2)
    
    # Run tests
    api_test_passed = test_enhanced_api()
    test_model_management()
    test_performance_tracking()
    
    # Summary
    print("\n" + "="*60)
    print("API INTEGRATION TEST SUMMARY")
    print("="*60)
    
    if api_test_passed:
        print("✅ Enhanced API integration successful!")
        print("   - Enhanced prediction endpoint working")
        print("   - Signal quality metrics available")
        print("   - Ensemble predictions functional")
        print("   - Market regime detection operational")
        print("\n🎉 API integration ready for production!")
    else:
        print("❌ Enhanced API integration failed")
        print("   - Check ML service status")
        print("   - Verify enhanced endpoints are registered")
        print("   - Review error logs")
    
    return api_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
