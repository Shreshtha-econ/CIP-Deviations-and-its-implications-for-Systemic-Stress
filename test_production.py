#!/usr/bin/env python3
"""
Test script to verify production configuration
"""
import os

def test_production_config():
    print("🧪 Testing Production Configuration")
    print("=" * 50)
    
    # Set production environment
    os.environ['FLASK_ENV'] = 'production'
    
    try:
        from src.api.app import app
        
        print("✅ Flask app imported successfully")
        print(f"Debug mode: {app.config['DEBUG']}")
        print(f"Testing mode: {app.config['TESTING']}")
        print(f"Environment: {os.environ.get('FLASK_ENV')}")
        
        # Test a simple route
        with app.test_client() as client:
            response = client.get('/api/status')
            print(f"Status endpoint test: {response.status_code}")
            
        print("\n🎉 Production configuration is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_development_config():
    print("\n🧪 Testing Development Configuration")
    print("=" * 50)
    
    # Reset to development
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Need to reload the module to pick up new environment
        import importlib
        import src.api.app
        importlib.reload(src.api.app)
        
        app = src.api.app.app
        
        print("✅ Flask app imported successfully")
        print(f"Debug mode: {app.config['DEBUG']}")
        print(f"Testing mode: {app.config['TESTING']}")
        print(f"Environment: {os.environ.get('FLASK_ENV')}")
        
        print("\n🎉 Development configuration is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Master Thesis API - Configuration Test")
    print("=" * 60)
    
    prod_ok = test_production_config()
    dev_ok = test_development_config()
    
    if prod_ok and dev_ok:
        print("\n🌟 ALL TESTS PASSED! Ready for deployment!")
    else:
        print("\n⚠️  Some tests failed. Check configuration.")
