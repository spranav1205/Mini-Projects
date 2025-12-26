# Retirement Portfolio Monte Carlo Simulator

A comprehensive Python-based Monte Carlo simulation tool for retirement portfolio planning and analysis. This tool helps you evaluate different withdrawal strategies, assess risk, and plan for a sustainable retirement.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Understanding the Results](#understanding-the-results)

## Features

- **Multiple Withdrawal Strategies**: Fixed amount, percentage-based, and dynamic withdrawals
- **Monte Carlo Simulation**: Run thousands of scenarios to assess probability outcomes
- **Comprehensive Metrics**: Success probability, ruin risk, shortfall analysis, and more
- **Rich Visualizations**: Annual traces, histograms, withdrawal analysis, and risk heatmaps
- **Flexible Configuration**: Customize market returns, inflation, taxes, and withdrawal parameters
- **Export Results**: Save simulation data to CSV for further analysis

## Installation

### Requirements

- Python 3.7 or higher
- NumPy
- Pandas
- Matplotlib

### Setup

```bash
# Clone or download the repository
cd "Retirement MCMC"

# Install required packages
pip install -r requirements.txt

# Verify installation (optional but recommended)
python test_installation.py
```

### Project Structure

```
Retirement MCMC/
├── simulation.py           # Core Monte Carlo simulation engine
├── static_baseline.py      # Deterministic baseline calculator
├── metrics.py             # Analysis and visualization functions
├── run_simulation.py      # Run simulations from config files
├── examples.py            # Pre-configured example scenarios
├── test_installation.py   # Verify package installation
├── config_example.ini     # Example configuration file
├── requirements.txt       # Python dependencies
├── README.md             # This file (technical reference)
├── GUIDE.md              # Quick start and user guide
└── LICENSE               # MIT license
```

## Quick Start

### Method 1: Using Example Scenarios

The quickest way to get started is to run pre-configured examples:

```bash
# Run all three scenarios and compare
python examples.py compare

# Or run individual scenarios
python examples.py conservative
python examples.py balanced
python examples.py aggressive
```

### Method 2: Using Configuration File

Create a configuration file (or modify `config_example.ini`) and run:

```bash
python run_simulation.py config_example.ini
```

### Method 3: Direct Python Usage

Here's a simple example to get started:

```python
from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import plot_traces_annual, print_summary_report

# Define your retirement scenario
initial_params = {
    'market': PARAM_PRESETS['market']['balanced_60_40'],  # 60/40 stock/bond portfolio
    'inflation': PARAM_PRESETS['inflation']['base']       # Moderate inflation
}

corpus = 1e7  # Starting corpus: ₹1 crore

withdrawal_params = {
    'monthly_amount': 30000,       # ₹30,000 per month
    'annual_lump_sum': 0,          # No annual lump sum
    'annual_increment': 0.0,       # No annual increases
    'grace_period_months': 0       # Start withdrawals immediately
}

# Run simulation
results = simulate_and_measure(
    initial_params, 
    corpus, 
    'fixed',                       # Fixed withdrawal strategy
    withdrawal_params,
    years=25,                      # 25-year retirement
    months=0,
    runs=1000,                     # 1000 Monte Carlo runs
    tax_rate=0.125                 # 12.5% tax on withdrawals
)

# View results
print_summary_report(results, withdrawal_params)
plot_traces_annual(results['traces'], initial_params['inflation'])
```

## 📖 User Guide

### Step 1: Define Your Retirement Parameters

#### Market Return Parameters

Choose a preset or define custom parameters `[mean, std]`:

```python
# Available presets
PARAM_PRESETS['market'] = {
    'equity': [0.13, 0.20],              # Aggressive: 13% return, 20% volatility
    'optimistic_equity': [0.15, 0.15],   # Very aggressive
    'balanced_60_40': [0.09, 0.12],      # Moderate: 60% stocks, 40% bonds
    'conservative_30_70': [0.07, 0.08],  # Conservative: 30% stocks, 70% bonds
    'gov_bonds': [0.065, 0.05],          # Very conservative
    'cash': [0.045, 0.015],              # Ultra-conservative
}

# Or define custom parameters
custom_market = [0.10, 0.15]  # 10% expected return, 15% volatility
```

#### Inflation Parameters

```python
PARAM_PRESETS['inflation'] = {
    'low': [0.040, 0.012],      # 4% inflation, low volatility
    'base': [0.052, 0.020],     # 5.2% inflation (India historical avg)
    'high': [0.06, 0.030],      # 6% inflation, high volatility
}
```

#### Initial Corpus

Specify your starting retirement corpus:

```python
corpus = 1e7        # ₹1 crore
corpus = 10000000   # ₹1 crore (same as 1e7)
```

### Step 2: Choose a Withdrawal Strategy

#### Strategy 1: Fixed Monthly Withdrawal

Best for predictable expenses with inflation adjustment:

```python
withdrawal_params = {
    'monthly_amount': 30000,           # Base monthly withdrawal
    'annual_lump_sum': 100000,         # Annual expense (e.g., vacation, insurance)
    'annual_increment': 0.02,          # 2% annual increase (beyond inflation)
    'grace_period_months': 6           # Wait 6 months before first withdrawal
}

results = simulate_and_measure(
    initial_params, corpus, 'fixed', withdrawal_params, 
    years=30, months=0, runs=1000
)
```

**When to use**: You have fixed expenses and want withdrawals to keep pace with inflation automatically.

#### Strategy 2: Percentage-Based Withdrawal

Withdraws a fixed percentage of current corpus:

```python
withdrawal_params = {
    'annual_percentage': 4.0  # Withdraw 4% annually (0.33% monthly)
}

results = simulate_and_measure(
    initial_params, corpus, 'percentage', withdrawal_params,
    years=30, months=0, runs=1000
)
```

**When to use**: You want withdrawals that adjust naturally with market performance (higher when markets are up, lower when down).

#### Strategy 3: Dynamic Withdrawal

Flexible withdrawal with floor and ceiling:

```python
withdrawal_params = {
    'min_withdrawal': 80000,       # Never withdraw less than ₹80k/month
    'max_withdrawal': 200000,      # Never withdraw more than ₹2 lakhs/month
    'target_percentage': 4.0       # Target 4% annual withdrawal
}

results = simulate_and_measure(
    initial_params, corpus, 'dynamic', withdrawal_params,
    years=30, months=0, runs=1000
)
```

**When to use**: You want flexibility with guardrails to prevent excessive withdrawals or insufficient income.

### Step 3: Run Simulations

```python
# Basic simulation
results = simulate_and_measure(
    initial_params,
    corpus,
    withdrawal_strategy,
    withdrawal_params,
    years=25,              # Retirement duration
    months=0,              # Additional months
    runs=1000,             # Number of Monte Carlo runs
    confidence=0.90,       # 90% confidence interval
    tax_rate=0.125         # 12.5% tax rate
)
```

### Step 4: Analyze Results

#### View Summary Report

```python
from metrics import print_summary_report

print_summary_report(results, withdrawal_params)
```

Output includes:
- Final corpus statistics (mean, median, percentiles)
- Confidence intervals
- Probability of beating inflation
- Ruin probability
- Shortfall risk
- Sustainability metrics

#### Visualize Results

```python
from metrics import (
    plot_traces_annual,
    plot_corpus_histogram,
    plot_withdrawal_analysis
)

# Annual corpus trajectory
plot_traces_annual(results['traces'], initial_params['inflation'])

# Corpus distribution at specific year
plot_corpus_histogram(results['traces'], year=10, 
                     inflation_params=initial_params['inflation'],
                     initial_corpus=corpus)

# Withdrawal sustainability analysis
plot_withdrawal_analysis(results['traces'], 
                        initial_params['market'],
                        initial_params['inflation'],
                        withdrawal_params)
```

### Export Results

#### Excel Export

Export comprehensive results to Excel with multiple sheets:

```python
from metrics import export_results_to_excel

# Export Monte Carlo results
export_results_to_excel(results, withdrawal_params, 'my_retirement_plan.xlsx')

# Export baseline calculator results
from static_baseline import StaticRetirementCalculator

calc = StaticRetirementCalculator(
    market_return=0.09, inflation_rate=0.052,
    corpus=1e7, withdrawal_strategy='fixed',
    withdrawal_params={'monthly_amount': 30000},
    tax_rate=0.125
)
calc.calculate(years=30)
calc.export_to_excel('baseline_plan.xlsx')
```

Excel exports include:
- **Monte Carlo**: Summary statistics, percentile traces, final corpus distribution
- **Baseline**: Monthly data, yearly summary, configuration details

#### Save Plots to Files

Save all visualization plots as high-resolution PNG files:

```python
from metrics import save_all_plots

# Save all plots to 'plots' directory
save_all_plots(results, initial_params['inflation'], 
               withdrawal_params, output_dir='plots', prefix='retirement_')

# Creates: retirement_traces_annual.png, retirement_corpus_histogram.png, etc.
```

## Advanced Usage

### Scenario Analysis

Compare different strategies side-by-side:

```python
strategies = {
    'Conservative': {
        'market': PARAM_PRESETS['market']['conservative_30_70'],
        'inflation': PARAM_PRESETS['inflation']['high'],
        'withdrawal': {'monthly_amount': 20000, 'annual_lump_sum': 0}
    },
    'Balanced': {
        'market': PARAM_PRESETS['market']['balanced_60_40'],
        'inflation': PARAM_PRESETS['inflation']['base'],
        'withdrawal': {'monthly_amount': 30000, 'annual_lump_sum': 0}
    },
    'Aggressive': {
        'market': PARAM_PRESETS['market']['equity'],
        'inflation': PARAM_PRESETS['inflation']['low'],
        'withdrawal': {'monthly_amount': 40000, 'annual_lump_sum': 0}
    }
}

for name, params in strategies.items():
    print(f"\n{'='*50}\nScenario: {name}\n{'='*50}")
    
    initial_params = {
        'market': params['market'],
        'inflation': params['inflation']
    }
    
    results = simulate_and_measure(
        initial_params, corpus, 'fixed', params['withdrawal'],
        years=25, months=0, runs=1000
    )
    
    print_summary_report(results, params['withdrawal'])
```

### Static Baseline Comparison

Compare Monte Carlo results against a deterministic baseline to understand the impact of volatility:

```python
from static_baseline import StaticRetirementCalculator, compare_with_baseline, print_comparison

# Run Monte Carlo simulation
mc_results = simulate_and_measure(
    {'market': PARAM_PRESETS['market']['balanced_60_40'],
     'inflation': PARAM_PRESETS['inflation']['base']},
    corpus=1e7,
    withdrawal_strategy='fixed',
    withdrawal_params={'monthly_amount': 30000},
    years=30, months=0, runs=1000
)

# Calculate static baseline (assumes constant returns)
static_calc = StaticRetirementCalculator(
    initial_corpus=1e7,
    annual_return=0.09,     # 9% constant
    annual_inflation=0.052,  # 5.2% constant
    monthly_withdrawal=30000,
    tax_rate=0.10
)
static_results = static_calc.calculate(years=30)

# Compare and analyze
comparison = compare_with_baseline(mc_results, static_results)
print_comparison(comparison)
```

The static baseline helps you understand:
- How much volatility affects your retirement success
- The "penalty" of market uncertainty vs. constant returns
- Whether Monte Carlo projections are conservative or aggressive

### Sensitivity Analysis

Test how changes in parameters affect outcomes:

```python
from metrics import plot_risk_heatmap

param_ranges = {
    'market': [
        [0.07, 0.08],   # Conservative
        [0.09, 0.12],   # Balanced
        [0.13, 0.20]    # Aggressive
    ],
    'inflation': [
        [0.04, 0.012],  # Low
        [0.052, 0.02],  # Base
        [0.06, 0.03]    # High
    ]
}

plot_risk_heatmap(param_ranges, corpus, withdrawal_params, years=25, runs=500)
```

### Monte Carlo with Different Tax Rates

```python
tax_rates = [0.10, 0.125, 0.15, 0.20, 0.30]

for rate in tax_rates:
    results = simulate_and_measure(
        initial_params, corpus, 'fixed', withdrawal_params,
        years=25, months=0, runs=1000, tax_rate=rate
    )
    
    ci_lo, ci_hi = results['confidence_interval']
    print(f"Tax Rate {rate*100}%: Final corpus ₹{ci_lo/1e7:.2f} - ₹{ci_hi/1e7:.2f} cr")
```

### Export Results for Excel Analysis

```python
from metrics import export_results_to_csv

export_results_to_csv(results, filename="my_retirement_plan.csv")
```

## Understanding the Results

### Key Metrics Explained

1. **Probability of Beating Inflation**: Percentage of scenarios where your final corpus exceeds the inflation-adjusted initial value. Aim for >70%.

2. **Ruin Probability**: Chance your portfolio runs out completely. Should be near 0% for conservative plans.

3. **Shortfall Risk**: Probability and magnitude of falling short of your inflation-adjusted target.

4. **Confidence Intervals**: Range where your final corpus will likely fall. E.g., 90% CI means 90% of scenarios fall within this range.

5. **Sustainability Metrics**: 
   - **Depletion probability**: Chance portfolio runs out during retirement
   - **Years to depletion**: Average time until portfolio exhaustion (if it occurs)

### Interpreting Visualizations

#### Annual Traces Plot
- **Blue lines**: Individual simulation paths
- **Dark blue line**: Median outcome (50th percentile)
- **Orange dashed**: Inflation-adjusted baseline
- **Shaded area**: 10th-90th percentile range (80% of outcomes)

**Good signs**: Median above inflation line, narrow confidence bands

#### Corpus Histogram
- Shows distribution of outcomes at a specific year
- Look for: Where most mass is relative to inflation baseline

#### Withdrawal Analysis
- **Green**: Sustainable withdrawal capacity
- **Red**: Your requested withdrawal
- Red above green = unsustainable (portfolio declining)
- Red below green = conservative (portfolio growing)

### Rules of Thumb

1. **Safe Withdrawal Rate**: 
   - Conservative: 3-3.5% of initial corpus annually
   - Moderate: 4% (classic rule)
   - Aggressive: 4.5-5%

2. **Success Rate Goals**:
   - 90%+: Very safe
   - 80-90%: Moderately safe
   - 70-80%: Acceptable with flexibility
   - <70%: Consider reducing withdrawals or increasing corpus

3. **Portfolio Allocation by Age**:
   - 60 years: 60-70% stocks, 30-40% bonds
   - 70 years: 40-50% stocks, 50-60% bonds
   - 80 years: 30-40% stocks, 60-70% bonds

## Examples

### Example 1: Conservative Retirement (70 years old, ₹3 crores)

```python
from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import print_summary_report, plot_traces_annual

# Conservative allocation for older retiree
initial_params = {
    'market': PARAM_PRESETS['market']['conservative_30_70'],
    'inflation': PARAM_PRESETS['inflation']['high']  # Assume worst case
}

corpus = 1e7  # ₹1 crore
withdrawal_params = {
    'monthly_amount': 30000,     # ₹30k/month (3.6% withdrawal rate)
    'annual_lump_sum': 100000,   # ₹1L annual
    'annual_increment': 0.0,
    'grace_period_months': 0
}

results = simulate_and_measure(
    initial_params, corpus, 'fixed', withdrawal_params,
    years=20, months=0, runs=1000, tax_rate=0.125
)

print_summary_report(results, withdrawal_params)
plot_traces_annual(results['traces'], initial_params['inflation'])
```

### Example 2: Moderate Retirement (60 years old, ₹5 crores)

```python
# Balanced allocation for younger retiree
initial_params = {
    'market': PARAM_PRESETS['market']['balanced_60_40'],
    'inflation': PARAM_PRESETS['inflation']['base']
}

corpus = 1e7  # ₹1 crore
withdrawal_params = {
    'monthly_amount': 30000,    # ₹30k/month (3.6% withdrawal rate)
    'annual_lump_sum': 100000,   # ₹1L annual for travel
    'annual_increment': 0.01,    # 1% real increase annually
    'grace_period_months': 0
}

results = simulate_and_measure(
    initial_params, corpus, 'fixed', withdrawal_params,
    years=30, months=0, runs=1000, tax_rate=0.10
)

print_summary_report(results, withdrawal_params)
```

### Example 3: Aggressive with Percentage Strategy (55 years old, ₹1 crore)

```python
# Aggressive allocation for early retiree
initial_params = {
    'market': PARAM_PRESETS['market']['equity'],
    'inflation': PARAM_PRESETS['inflation']['base']
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
```

### Example 4: Dynamic Strategy with Guardrails

```python
# Flexible strategy with protection
initial_params = {
    'market': PARAM_PRESETS['market']['balanced_60_40'],
    'inflation': PARAM_PRESETS['inflation']['base']
}

corpus = 1e7  # ₹1 crore
withdrawal_params = {
    'min_withdrawal': 20000,    # Minimum ₹20k/month
    'max_withdrawal': 50000,    # Maximum ₹50k/month
    'target_percentage': 4.0     # Target 4% annually
}

results = simulate_and_measure(
    initial_params, corpus, 'dynamic', withdrawal_params,
    years=30, months=0, runs=1000, tax_rate=0.125
)

print_summary_report(results, withdrawal_params)
```

## API Reference

### Core Functions

#### `simulate_and_measure()`
Main function to run Monte Carlo simulations.

**Parameters:**
- `initial_params` (dict): Market and inflation parameters
- `corpus` (float): Initial portfolio value
- `withdrawal_strategy` (str): 'fixed', 'percentage', or 'dynamic'
- `withdrawal_params` (dict): Strategy-specific parameters
- `years` (int): Number of years to simulate
- `months` (int): Additional months
- `runs` (int): Number of Monte Carlo runs
- `confidence` (float): Confidence level (default 0.90)
- `tax_rate` (float): Tax rate on withdrawals (default 0.125)

**Returns:** Dictionary with 'traces', 'final_corpuses', 'confidence_interval', 'initial_params', 'initial_corpus'

### Metrics Functions

See [metrics.py](metrics.py) for complete documentation of:
- `print_summary_report()`: Comprehensive text report
- `plot_traces_annual()`: Annual trajectory visualization
- `plot_corpus_histogram()`: Distribution at specific year
- `plot_withdrawal_analysis()`: Sustainable vs requested withdrawals
- `probability_beat_inflation()`: Calculate inflation-beating probability
- `calculate_ruin_probability()`: Portfolio depletion risk
- `calculate_shortfall_risk()`: Shortfall probability and magnitude
- `export_results_to_csv()`: Export data for external analysis

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and planning purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.

---

## Changelog

### Version 2.0 - December 26, 2025

**Major Refactoring**: Complete transformation from research code to production-ready package

**New Features**:
- Modular architecture (simulation, metrics, config support, examples)
- Deterministic baseline calculator (static_baseline.py)
- Excel export for both Monte Carlo and baseline results
- Graph export with save_all_plots()
- Three withdrawal strategies (fixed, percentage, dynamic)
- Parameter presets (6 market, 3 inflation scenarios)
- Comprehensive metrics (success, ruin, shortfall, sustainability)
- Config file support (.ini format)
- Automated test suite

**Breaking Changes**:
- New parameter structure for withdrawal_params
- tax_rate now configurable parameter (not hardcoded)
- Function signatures updated for clarity

**Documentation**:
- README.md: Complete technical reference
- GUIDE.md: Quick start and user guide
- Fully commented examples and config files

---

## Support

For questions or issues, see GUIDE.md for common scenarios and troubleshooting.

---
