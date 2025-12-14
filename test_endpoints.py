#!/usr/bin/env python
"""
Test script to verify Green AI endpoints and functionality
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_endpoints():
    """Test all main endpoints"""
    
    print("\n" + "="*70)
    print("GREEN AI ENDPOINT TESTS")
    print("="*70 + "\n")
    
    # Test 1: Home page
    print("1. Testing Home Page...")
    try:
        resp = requests.get(f"{BASE_URL}/")
        print(f"   Status: {resp.status_code}")
        if "Green AI" in resp.text:
            print("   ✓ Home page loads with content")
        else:
            print("   ✗ Home page missing content")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Configuration API
    print("\n2. Testing Configuration API...")
    try:
        resp = requests.get(f"{BASE_URL}/api/config")
        data = resp.json()
        print(f"   Status: {resp.status_code}")
        if data.get('status') == 'success':
            print(f"   ✓ Config loaded")
            config = data.get('data', {})
            print(f"     - Dataset: {config.get('dataset')}")
            print(f"     - Population: {config.get('population_size')}")
            print(f"     - Generations: {config.get('generations')}")
        else:
            print(f"   ✗ Config failed: {data.get('message')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Configuration Validation
    print("\n3. Testing Configuration Validation...")
    try:
        resp = requests.get(f"{BASE_URL}/api/config/validate")
        data = resp.json()
        print(f"   Status: {resp.status_code}")
        is_valid = data.get('data', {}).get('valid')
        if is_valid:
            print("   ✓ Configuration is valid")
        else:
            print(f"   ✗ Configuration invalid: {data.get('message')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: GA Status
    print("\n4. Testing GA Status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/ga/status")
        data = resp.json()
        print(f"   Status: {resp.status_code}")
        ga_state = data.get('data', {})
        print(f"   ✓ GA Status retrieved")
        print(f"     - Running: {ga_state.get('running')}")
        print(f"     - Progress: {ga_state.get('progress')}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Optimizer page
    print("\n5. Testing Optimizer Page...")
    try:
        resp = requests.get(f"{BASE_URL}/optimizer")
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   ✓ Optimizer page loads")
        else:
            print(f"   ✗ Optimizer page failed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 6: Dashboard page
    print("\n6. Testing Dashboard Page...")
    try:
        resp = requests.get(f"{BASE_URL}/dashboard")
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   ✓ Dashboard page loads")
        else:
            print(f"   ✗ Dashboard page failed")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 7: Static files
    print("\n7. Testing Static Files...")
    try:
        resp = requests.get(f"{BASE_URL}/static/css/style.css")
        print(f"   CSS Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   ✓ CSS loads correctly")
        else:
            print(f"   ✗ CSS failed: {resp.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        resp = requests.get(f"{BASE_URL}/static/js/utils.js")
        print(f"   JS Status: {resp.status_code}")
        if resp.status_code == 200:
            print("   ✓ JavaScript loads correctly")
        else:
            print(f"   ✗ JavaScript failed: {resp.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == '__main__':
    print("Waiting for Flask server to start (5 seconds)...")
    time.sleep(5)
    test_endpoints()
