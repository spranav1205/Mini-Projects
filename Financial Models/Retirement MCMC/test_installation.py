"""
Test script to verify the retirement simulation package works correctly.

Run this to ensure all modules are properly installed and functioning.
"""

import sys


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        print("  [OK] NumPy imported successfully")
    except ImportError:
        print("  ✗ NumPy import failed - run: pip install numpy")
        return False
    
    try:
        import pandas as pd
        print("  [OK] Pandas imported successfully")
    except ImportError:
        print("  ✗ Pandas import failed - run: pip install pandas")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  [OK] Matplotlib imported successfully")
    except ImportError:
        print("  ✗ Matplotlib import failed - run: pip install matplotlib")
        return False
    
    try:
        from simulation import simulate_and_measure, PARAM_PRESETS, RetirementMCMC
        print("  [OK] simulation.py imported successfully")
    except ImportError as e:
        print(f"  ✗ simulation.py import failed: {e}")
        return False
    
    try:
        from metrics import (
            print_summary_report,
            plot_traces_annual,
            probability_beat_inflation
        )
        print("  [OK] metrics.py imported successfully")
    except ImportError as e:
        print(f"  ✗ metrics.py import failed: {e}")
        return False
    
    return True


def test_basic_simulation():
    """Test that a basic simulation runs without errors."""
    print("\nTesting basic simulation...")
    try:
        from simulation import simulate_and_measure, PARAM_PRESETS
        import numpy as np
        
        # Simple test scenario
        initial_params = {
            'market': PARAM_PRESETS['market']['balanced_60_40'],
            'inflation': PARAM_PRESETS['inflation']['base']
        }
        
        corpus = 1e7  # 1 crore
        withdrawal_params = {
            'monthly_amount': 50000,
            'annual_lump_sum': 0,
            'annual_increment': 0.0,
            'grace_period_months': 0
        }
        
        # Run small simulation
        results = simulate_and_measure(
            initial_params, corpus, 'fixed', withdrawal_params,
            years=5, months=0, runs=10  # Small test
        )
        
        # Verify results
        assert 'traces' in results
        assert 'final_corpuses' in results
        assert 'confidence_interval' in results
        assert len(results['traces']) == 10
        assert len(results['final_corpuses']) == 10
        
        print("  [OK] Basic simulation completed successfully")
        print(f"  [OK] Generated {len(results['traces'])} traces")
        print(f"  [OK] Mean final corpus: ₹{np.mean(results['final_corpuses'])/1e5:.2f} lakhs")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_percentage_strategy():
    """Test percentage-based withdrawal strategy."""
    print("\nTesting percentage strategy...")
    try:
        from simulation import simulate_and_measure, PARAM_PRESETS
        
        initial_params = {
            'market': PARAM_PRESETS['market']['balanced_60_40'],
            'inflation': PARAM_PRESETS['inflation']['base']
        }
        
        corpus = 1e7
        withdrawal_params = {
            'annual_percentage': 4.0
        }
        
        results = simulate_and_measure(
            initial_params, corpus, 'percentage', withdrawal_params,
            years=5, months=0, runs=10
        )
        
        assert len(results['traces']) == 10
        print("  [OK] Percentage strategy works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Percentage strategy failed: {e}")
        return False


def test_dynamic_strategy():
    """Test dynamic withdrawal strategy."""
    print("\nTesting dynamic strategy...")
    try:
        from simulation import simulate_and_measure, PARAM_PRESETS
        
        initial_params = {
            'market': PARAM_PRESETS['market']['balanced_60_40'],
            'inflation': PARAM_PRESETS['inflation']['base']
        }
        
        corpus = 1e7
        withdrawal_params = {
            'min_withdrawal': 30000,
            'max_withdrawal': 100000,
            'target_percentage': 4.0
        }
        
        results = simulate_and_measure(
            initial_params, corpus, 'dynamic', withdrawal_params,
            years=5, months=0, runs=10
        )
        
        assert len(results['traces']) == 10
        print("  [OK] Dynamic strategy works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Dynamic strategy failed: {e}")
        return False


def test_metrics():
    """Test metrics and analysis functions."""
    print("\nTesting metrics functions...")
    try:
        from simulation import simulate_and_measure, PARAM_PRESETS
        from metrics import (
            probability_beat_inflation,
            calculate_ruin_probability,
            calculate_shortfall_risk,
            annualize_traces
        )
        
        # Generate test data
        initial_params = {
            'market': PARAM_PRESETS['market']['balanced_60_40'],
            'inflation': PARAM_PRESETS['inflation']['base']
        }
        
        corpus = 1e7
        withdrawal_params = {
            'monthly_amount': 40000,
            'annual_lump_sum': 0,
            'annual_increment': 0.0,
            'grace_period_months': 0
        }
        
        results = simulate_and_measure(
            initial_params, corpus, 'fixed', withdrawal_params,
            years=5, months=0, runs=10
        )
        
        # Test probability_beat_inflation
        prob, baseline = probability_beat_inflation(
            corpus, results['final_corpuses'], 
            initial_params['inflation'], 5
        )
        assert 0 <= prob <= 1
        print(f"  [OK] Probability beat inflation: {prob*100:.1f}%")
        
        # Test ruin probability
        ruin_prob = calculate_ruin_probability(results['final_corpuses'])
        assert 0 <= ruin_prob <= 1
        print(f"  [OK] Ruin probability: {ruin_prob*100:.1f}%")
        
        # Test shortfall risk
        shortfall_prob, avg_shortfall = calculate_shortfall_risk(
            results['final_corpuses'], baseline
        )
        assert 0 <= shortfall_prob <= 1
        print(f"  [OK] Shortfall probability: {shortfall_prob*100:.1f}%")
        
        # Test annualize_traces
        annual, n_years = annualize_traces(results['traces'])
        assert n_years == 5
        assert annual.shape == (10, 5)
        print("  [OK] Trace annualization works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("RETIREMENT SIMULATION PACKAGE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Simulation", test_basic_simulation),
        ("Percentage Strategy", test_percentage_strategy),
        ("Dynamic Strategy", test_dynamic_strategy),
        ("Metrics Functions", test_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:25s} {status}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\nAll tests passed! Your installation is working correctly.")
        print("\nNext steps:")
        print("  1. Try: python examples.py balanced")
        print("  2. Or:  python run_simulation.py config_example.ini")
        return True
    else:
        print("\nSome tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Ensure you're in the correct directory")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
