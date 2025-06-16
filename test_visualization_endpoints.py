"""
Visualization Endpoints Test Script
Tests all chart generation endpoints to ensure they work correctly.
"""

import requests
import json
import base64
import os
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5050"
TEST_OUTPUT_DIR = "test_charts_output"

def create_output_directory():
    """Create output directory for test charts."""
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
        print(f"✅ Created output directory: {TEST_OUTPUT_DIR}")

def save_chart_image(image_base64, filename):
    """Save base64 image to file."""
    try:
        image_data = base64.b64decode(image_base64)
        filepath = os.path.join(TEST_OUTPUT_DIR, f"{filename}.png")
        with open(filepath, 'wb') as f:
            f.write(image_data)
        print(f"✅ Saved chart: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {filename}: {str(e)}")
        return False

def test_endpoint(endpoint, description, save_name=None):
    """Test a single visualization endpoint."""
    print(f"\n🔄 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print(f"✅ SUCCESS: {description}")
                
                # Save chart if image data is available
                if save_name and 'image' in data.get('data', {}):
                    save_chart_image(data['data']['image'], save_name)
                elif save_name and 'images' in data.get('data', {}):
                    # Handle multiple images (e.g., cip_deviation_vs_band)
                    images = data['data']['images']
                    for currency, image_data in images.items():
                        save_chart_image(image_data, f"{save_name}_{currency}")
                
                return True
            else:
                print(f"❌ FAILED: {description} - {data.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ FAILED: {description} - HTTP {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: {description} - Connection error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ FAILED: {description} - {str(e)}")
        return False

def test_html_endpoint(endpoint, description):
    """Test HTML visualization endpoints."""
    print(f"\n🔄 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        
        if response.status_code == 200:
            if 'text/html' in response.headers.get('content-type', ''):
                print(f"✅ SUCCESS: {description} - HTML page generated")
                
                # Save HTML file
                filename = os.path.join(TEST_OUTPUT_DIR, f"{description.replace(' ', '_').lower()}.html")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"✅ Saved HTML: {filename}")
                return True
            else:
                print(f"❌ FAILED: {description} - Not HTML content")
                return False
        else:
            print(f"❌ FAILED: {description} - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {description} - {str(e)}")
        return False

def check_api_status():
    """Check if API is running and accessible."""
    print("🔄 Checking API status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running - Status: {data.get('data', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API not accessible - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not accessible - {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 VISUALIZATION ENDPOINTS TEST SUITE")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API Base URL: {API_BASE_URL}")
    
    # Create output directory
    create_output_directory()
    
    # Check API status
    if not check_api_status():
        print("\n❌ CRITICAL: API is not running or not accessible")
        print("   Please start the API first: python scripts/start_api.py")
        return False
    
    # Test visualization endpoints
    tests = [
        ("/api/charts/cip_deviations", "CIP Deviations Chart", "cip_deviations"),
        ("/api/charts/bandwidth_volatility", "Bandwidth vs Volatility Chart", "bandwidth_volatility"),
        ("/api/charts/bandwidth_volatility?currency=GBP", "Bandwidth vs Volatility (GBP)", "bandwidth_volatility_gbp"),
        ("/api/charts/cip_deviation_vs_band", "CIP Deviation vs Band Charts", "cip_deviation_vs_band"),
        ("/api/charts/ciss_indicator", "CISS Indicator Chart", "ciss_indicator"),
        ("/api/charts/ciss_comparison", "CISS Comparison Chart", "ciss_comparison"),
        ("/api/charts/cross_correlation", "Cross-Correlation Chart", "cross_correlation"),
        ("/api/charts/summary_dashboard", "Summary Dashboard", "summary_dashboard"),
    ]
    
    # Test HTML endpoints
    html_tests = [
        ("/charts/cip_deviations_view", "CIP Deviations HTML View"),
        ("/charts/dashboard_view", "Dashboard HTML View"),
    ]
    
    # Run tests
    successful_tests = 0
    total_tests = len(tests) + len(html_tests)
    
    print(f"\n🚀 Running {len(tests)} visualization endpoint tests...")
    for endpoint, description, save_name in tests:
        if test_endpoint(endpoint, description, save_name):
            successful_tests += 1
        time.sleep(1)  # Small delay between tests
    
    print(f"\n🌐 Running {len(html_tests)} HTML endpoint tests...")
    for endpoint, description in html_tests:
        if test_html_endpoint(endpoint, description):
            successful_tests += 1
        time.sleep(1)
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✅ All visualization endpoints are working correctly!")
        print(f"📁 Charts saved to: {TEST_OUTPUT_DIR}/")
    else:
        print(f"\n⚠️  {total_tests - successful_tests} tests failed")
        print("   Check the error messages above for details")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
