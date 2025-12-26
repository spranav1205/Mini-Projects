"""
Simple example demonstrating retirement portfolio simulation.

This script shows three common retirement scenarios:
1. Conservative (older retiree, safe investments)
2. Balanced (middle-age retiree, moderate risk)
3. Aggressive (younger retiree, growth-focused)
"""

from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import print_summary_report, plot_traces_annual
import matplotlib.pyplot as plt


def scenario_conservative():
    """Conservative scenario: 70 years old, ₹1 crore, 20-year horizon."""
    print("\n" + "="*70)
    print("SCENARIO 1: CONSERVATIVE RETIREMENT")
    print("Age: 70 | Corpus: ₹1 crore | Horizon: 20 years")
    print("="*70)
    
    initial_params = {
        'market': PARAM_PRESETS['market']['conservative_30_70'],  # 30% stocks, 70% bonds
        'inflation': PARAM_PRESETS['inflation']['high']           # Worst-case inflation
    }
    
    corpus = 1e7  # ₹1 crore
    withdrawal_params = {
        'monthly_amount': 30000,       # ₹30,000/month
        'annual_lump_sum': 100000,     # ₹1 lakh annual
        'annual_increment': 0.0,       # No real increase
        'grace_period_months': 0
    }
    
    results = simulate_and_measure(
        initial_params, corpus, 'fixed', withdrawal_params,
        years=20, months=0, runs=1000, tax_rate=0.125
    )
    
    print_summary_report(results, withdrawal_params)
    plot_traces_annual(results['traces'], initial_params['inflation'])
    
    return results


def scenario_balanced():
    """Balanced scenario: 60 years old, ₹1 crore, 30-year horizon."""
    print("\n" + "="*70)
    print("SCENARIO 2: BALANCED RETIREMENT")
    print("Age: 60 | Corpus: ₹1 crore | Horizon: 30 years")
    print("="*70)
    
    initial_params = {
        'market': PARAM_PRESETS['market']['balanced_60_40'],  # 60% stocks, 40% bonds
        'inflation': PARAM_PRESETS['inflation']['base']       # Base inflation
    }
    
    corpus = 1e7  # ₹1 crore
    withdrawal_params = {
        'monthly_amount': 30000,       # ₹30,000/month
        'annual_lump_sum': 100000,     # ₹1 lakh annual
        'annual_increment': 0.01,      # 1% real increase
        'grace_period_months': 0
    }
    
    results = simulate_and_measure(
        initial_params, corpus, 'fixed', withdrawal_params,
        years=30, months=0, runs=1000, tax_rate=0.10
    )
    
    print_summary_report(results, withdrawal_params)
    plot_traces_annual(results['traces'], initial_params['inflation'])
    plt.show()
    
    return results


def scenario_aggressive():
    """Aggressive scenario: 55 years old, ₹1 crore, 35-year horizon."""
    print("\n" + "="*70)
    print("SCENARIO 3: AGGRESSIVE RETIREMENT")
    print("Age: 55 | Corpus: ₹1 crore | Horizon: 35 years")
    print("="*70)
    
    initial_params = {
        'market': PARAM_PRESETS['market']['equity'],      # Equity-heavy
        'inflation': PARAM_PRESETS['inflation']['base']   # Base inflation
    }
    
    corpus = 1e7  # ₹1 crore
    withdrawal_params = {
        'annual_percentage': 4.5  # 4.5% variable withdrawal
    }
    
    results = simulate_and_measure(
        initial_params, corpus, 'percentage', withdrawal_params,
        years=35, months=0, runs=1000, tax_rate=0.10
    )
    
    print_summary_report(results, withdrawal_params)
    plot_traces_annual(results['traces'], initial_params['inflation'])
    plt.show()
    
    return results


def compare_scenarios():
    """Run and compare all three scenarios."""
    print("\n" + "="*70)
    print("RETIREMENT SCENARIO COMPARISON")
    print("="*70)
    
    # Run all scenarios
    results_conservative = scenario_conservative()
    results_balanced = scenario_balanced()
    results_aggressive = scenario_aggressive()
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    scenarios = [
        ("Conservative", results_conservative),
        ("Balanced", results_balanced),
        ("Aggressive", results_aggressive)
    ]
    
    print("\nFinal Corpus Comparison (Median):")
    for name, results in scenarios:
        import numpy as np
        median = np.median(results['final_corpuses'])
        initial = results['initial_corpus']
        print(f"  {name:12s}: ₹{median/1e7:.2f} cr (started with ₹{initial/1e7:.2f} cr)")
    
    print("\nSuccess Rate (Beating Inflation):")
    for name, results in scenarios:
        from metrics import probability_beat_inflation
        initial = results['initial_corpus']
        years = len(results['traces'][0]) // 12
        prob, _ = probability_beat_inflation(
            initial, 
            results['final_corpuses'],
            results['initial_params']['inflation'],
            years
        )
        print(f"  {name:12s}: {prob*100:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run individual scenario or comparison
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1].lower()
        if scenario == 'conservative':
            scenario_conservative()
        elif scenario == 'balanced':
            scenario_balanced()
        elif scenario == 'aggressive':
            scenario_aggressive()
        elif scenario == 'compare':
            compare_scenarios()
        else:
            print(f"Unknown scenario: {scenario}")
            print("Available: conservative, balanced, aggressive, compare")
    else:
        # Default: run comparison
        compare_scenarios()
