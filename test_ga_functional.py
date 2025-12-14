#!/usr/bin/env python
"""
Test GA functionality - run a quick optimization to verify everything works
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:5000"

def test_ga_workflow():
    """Test complete GA optimization workflow"""
    
    print("\n" + "="*70)
    print("GREEN AI - GENETIC ALGORITHM FUNCTIONAL TEST")
    print("="*70 + "\n")
    
    # Step 1: Verify config
    print("Step 1: Verifying Configuration...")
    try:
        resp = requests.get(f"{BASE_URL}/api/config/validate")
        data = resp.json()
        if data.get('status') != 'success' or not data.get('data', {}).get('valid'):
            print("   ✗ Configuration invalid!")
            return False
        config = requests.get(f"{BASE_URL}/api/config").json().get('data', {})
        print(f"   ✓ Configuration valid")
        print(f"     Dataset: {config.get('dataset')}")
        print(f"     Population: {config.get('population_size')}")
        print(f"     Generations: {config.get('generations')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Step 2: Check GA status before running
    print("\nStep 2: Checking Initial GA Status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/ga/status")
        data = resp.json().get('data', {})
        print(f"   ✓ Initial status retrieved")
        print(f"     Running: {data.get('running')}")
        print(f"     Progress: {data.get('progress')}%")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Step 3: Start optimization
    print("\nStep 3: Starting Optimization...")
    try:
        resp = requests.post(f"{BASE_URL}/api/ga/run")
        data = resp.json()
        if data.get('status') != 'success':
            print(f"   ✗ Failed to start: {data.get('message')}")
            return False
        print("   ✓ Optimization started successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Step 4: Monitor progress
    print("\nStep 4: Monitoring Progress (up to 3 minutes)...")
    print("   ", end="", flush=True)
    
    start_time = time.time()
    max_wait = 180  # 3 minutes
    last_progress = 0
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{BASE_URL}/api/ga/status")
            data = resp.json().get('data', {})
            
            current_progress = data.get('progress', 0)
            is_running = data.get('running', False)
            generation = data.get('generation', 0)
            
            # Show progress
            if current_progress != last_progress:
                print(f"\n   Gen {generation}/{config.get('generations')}: {current_progress}%", end="", flush=True)
                last_progress = current_progress
            else:
                print(".", end="", flush=True)
            
            # Check if done
            if not is_running and current_progress == 100:
                print("\n   ✓ Optimization completed!")
                break
            
            time.sleep(2)
        except Exception as e:
            print(f"\n   ✗ Error monitoring progress: {e}")
            return False
    else:
        print("\n   ✗ Optimization timed out (>3 minutes)")
        return False
    
    # Step 5: Get results
    print("\nStep 5: Retrieving Results...")
    try:
        resp = requests.get(f"{BASE_URL}/api/results")
        data = resp.json()
        if data.get('status') != 'success':
            print(f"   ✗ Could not retrieve results")
            return False
        
        results = data.get('data', {})
        print("   ✓ Results retrieved")
        print(f"     Best Accuracy: {results.get('best_accuracy', 'N/A')}")
        print(f"     Total Generations: {results.get('generations_completed', 0)}")
        print(f"     Total Individuals Evaluated: {results.get('total_evaluated', 0)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Step 6: Get best architecture
    print("\nStep 6: Retrieving Best Architecture...")
    try:
        resp = requests.get(f"{BASE_URL}/api/architecture")
        data = resp.json()
        if data.get('status') != 'success':
            print(f"   ✗ Could not retrieve architecture")
            return False
        
        arch = data.get('data', {})
        print("   ✓ Best architecture retrieved")
        print(f"     Architecture: {arch.get('architecture', [])}")
        print(f"     Fitness: {arch.get('fitness', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Final summary
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - GA FULLY FUNCTIONAL!")
    print("="*70)
    print("\nOptimization Summary:")
    print(f"  - Dataset: {config.get('dataset')}")
    print(f"  - Population: {config.get('population_size')}")
    print(f"  - Generations: {config.get('generations')}")
    print(f"  - Best Accuracy: {results.get('best_accuracy', 'N/A')}")
    print("\nThe genetic algorithm is working correctly!")
    print("Visit http://localhost:5000/dashboard to view detailed results.\n")
    
    return True

if __name__ == '__main__':
    success = test_ga_workflow()
    sys.exit(0 if success else 1)
