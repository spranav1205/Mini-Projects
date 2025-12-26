# Retirement Planning Simulator - Quick Start Guide

A comprehensive Monte Carlo simulation tool for retirement portfolio planning. This guide gets you started in 5 minutes.

---

## Installation

```bash
cd "Financial Models/Retirement MCMC"
pip install -r requirements.txt
python test_installation.py  # Verify setup
```

**Requirements**: Python 3.7+, numpy, pandas, matplotlib, openpyxl

---

## Three Ways to Use

### 1. Quick Examples (Easiest)

Run pre-built scenarios:

```bash
python examples.py conservative  # Safe, older retiree
python examples.py balanced      # Moderate risk
python examples.py aggressive    # Growth-focused
python examples.py compare       # Compare all three
```

### 2. Configuration Files (Flexible)

Create `my_plan.ini`:

```ini
[scenario]
name = My Retirement Plan
description = Conservative planning for age 65

[portfolio]
initial_corpus = 10000000  # ₹1 crore
market_preset = conservative_30_70
inflation_preset = high

[withdrawal]
strategy = fixed
monthly_amount = 30000    # ₹30k/month
annual_lump_sum = 100000  # ₹1L annually

[simulation]
years = 25
runs = 1000
```

Run it:
```bash
python run_simulation.py my_plan.ini
```

### 3. Python API (Full Control)

```python
from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import print_summary_report, export_results_to_excel, save_all_plots

# Setup
initial_params = {
    'market': PARAM_PRESETS['market']['balanced_60_40'],
    'inflation': PARAM_PRESETS['inflation']['base']
}

corpus = 1e7  # ₹1 crore
withdrawal_params = {'monthly_amount': 30000}

# Run simulation
results = simulate_and_measure(
    initial_params, corpus, 'fixed', withdrawal_params,
    years=30, runs=1000, tax_rate=0.125
)

# Analyze
print_summary_report(results, withdrawal_params)

# Export
export_results_to_excel(results, withdrawal_params, 'my_results.xlsx')
save_all_plots(results, initial_params['inflation'], withdrawal_params, 'plots')
```

---

## Key Features

### Core Simulation
- **Monte Carlo Analysis**: Run 100-10,000 scenarios with random market returns/inflation
- **Stochastic Engine**: Realistic modeling of market volatility and uncertainty
- **Deterministic Baseline**: Compare against constant-return scenarios

### Withdrawal Strategies

**Fixed Strategy** - Traditional approach:
- Monthly amount (inflation-adjusted)
- Optional annual lump sums
- Annual increments beyond inflation
- Grace period support

**Percentage Strategy** - Variable approach:
- Annual % of current corpus
- Automatically adjusts to performance

**Dynamic Strategy** - Guardrails approach:
- Target percentage with min/max limits
- Flexible but safe

### Parameter Presets

**Market Scenarios**:
- `optimistic_equity`: 14% return, 18% volatility
- `equity`: 12% return, 15% volatility
- `balanced_60_40`: 9% return, 10% volatility
- `conservative_30_70`: 7% return, 7% volatility
- `gov_bonds`: 6.5% return, 5% volatility
- `cash`: 5% return, 2% volatility

**Inflation Scenarios**:
- `low`: 4.0% ± 1.2%
- `base`: 5.2% ± 2.0%
- `high`: 6.0% ± 3.0%

### Analysis & Metrics

**Risk Metrics**:
- Success probability (beat inflation)
- Ruin probability (corpus exhaustion)
- Shortfall risk (fall short of target)
- Sustainability analysis

**Visualizations**:
- Annual corpus traces (median + percentiles)
- Final corpus distributions
- Withdrawal sustainability analysis
- Risk heatmaps

**Export Options**:
- Excel files (multi-sheet, formatted)
- High-resolution PNG graphs
- CSV data exports

---

## Common Scenarios

### Scenario A: Conservative (Age 65+)

**Profile**: Older retiree, needs stability, 20-year horizon

```ini
[portfolio]
initial_corpus = 10000000
market_preset = conservative_30_70  # 30% stocks, 70% bonds

[withdrawal]
strategy = fixed
monthly_amount = 30000
annual_lump_sum = 100000
```

**Expected Outcome**: ~85-90% success rate, low volatility

---

### Scenario B: Balanced (Age 55-65)

**Profile**: Mid-age retiree, moderate risk tolerance, 30-year horizon

```ini
[portfolio]
initial_corpus = 10000000
market_preset = balanced_60_40  # 60% stocks, 40% bonds

[withdrawal]
strategy = fixed
monthly_amount = 30000
```

**Expected Outcome**: ~75-80% success rate, moderate growth potential

---

### Scenario C: Aggressive (Age 45-55)

**Profile**: Early retiree, flexible spending, 35+ year horizon

```ini
[portfolio]
initial_corpus = 10000000
market_preset = equity  # Equity-heavy

[withdrawal]
strategy = percentage
annual_percentage = 4.5  # Variable based on corpus
```

**Expected Outcome**: Higher volatility, significant growth potential

---

## Understanding Results

### Success Metrics

**Success Probability**: Likelihood of beating inflation
- 80%+ = Excellent
- 70-80% = Good
- 60-70% = Moderate risk
- <60% = High risk, consider adjustments

**Ruin Probability**: Chance of complete depletion
- <5% = Safe
- 5-10% = Acceptable for most
- >10% = Risky, reduce withdrawals

**Withdrawal Rate**: Annual withdrawal / corpus
- 3-4% = Conservative (traditional "safe")
- 4-5% = Moderate
- 5-6% = Aggressive
- 6%+ = Very aggressive

### Adjusting Your Plan

**If results show high risk**:

1. **Reduce withdrawals**
   ```ini
   monthly_amount = 25000  # Instead of 30000
   ```

2. **Increase equity allocation** (if time horizon permits)
   ```ini
   market_preset = balanced_60_40  # Instead of conservative
   ```

3. **Plan for longer horizon**
   ```ini
   years = 35  # Add buffer years
   ```

**If results are very conservative**:

1. **Increase withdrawals gradually**
   ```ini
   monthly_amount = 35000
   annual_increment = 0.01  # 1% real increase annually
   ```

2. **Use dynamic strategy**
   ```ini
   strategy = dynamic
   target_percentage = 4.5
   min_withdrawal = 25000
   max_withdrawal = 60000
   ```

---

## Package Structure

```
Retirement MCMC/
├── simulation.py           # Monte Carlo simulation engine
├── static_baseline.py      # Deterministic calculator
├── metrics.py              # Analysis & visualization
├── run_simulation.py       # Config file executor
├── examples.py             # Pre-built scenarios
├── test_installation.py    # Validation suite
├── config_example.ini      # Configuration template
├── requirements.txt        # Dependencies
├── README.md              # Technical documentation
└── GUIDE.md               # This file (quick start)
```

---

## Pro Tips

1. **Test Worst-Case**: Use high inflation + conservative market returns
2. **Leave Safety Buffer**: Target 75-80% success, not barely 70%
3. **Plan for Longevity**: Add 5-10 years to life expectancy
4. **Include Medical Buffer**: Add ₹2-5L annual for healthcare
5. **Review Annually**: Update assumptions as markets/life changes
6. **Use Baseline Comparison**: Compare Monte Carlo vs deterministic
7. **Export Everything**: Keep Excel files for future reference

---

## Example Workflows

### Workflow 1: First-Time User

```bash
# 1. Verify installation
python test_installation.py

# 2. Run example
python examples.py balanced

# 3. Review results, then customize
cp config_example.ini my_retirement.ini
# Edit my_retirement.ini with your numbers

# 4. Run your plan
python run_simulation.py my_retirement.ini
```

### Workflow 2: Scenario Comparison

```python
from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import print_summary_report

scenarios = {
    'Conservative': {
        'market': PARAM_PRESETS['market']['conservative_30_70'],
        'withdrawal': 20000
    },
    'Moderate': {
        'market': PARAM_PRESETS['market']['balanced_60_40'],
        'withdrawal': 30000
    },
    'Aggressive': {
        'market': PARAM_PRESETS['market']['equity'],
        'withdrawal': 40000
    }
}

for name, params in scenarios.items():
    print(f"\n{'='*50}\n{name} Scenario\n{'='*50}")
    
    results = simulate_and_measure(
        {'market': params['market'], 
         'inflation': PARAM_PRESETS['inflation']['base']},
        corpus=1e7,
        withdrawal_strategy='fixed',
        withdrawal_params={'monthly_amount': params['withdrawal']},
        years=25, runs=1000
    )
    
    print_summary_report(results, {'monthly_amount': params['withdrawal']})
```

### Workflow 3: Baseline Comparison

```python
from simulation import simulate_and_measure, PARAM_PRESETS
from static_baseline import StaticRetirementCalculator, compare_with_baseline, print_comparison

# Monte Carlo
mc_results = simulate_and_measure(
    {'market': PARAM_PRESETS['market']['balanced_60_40'],
     'inflation': PARAM_PRESETS['inflation']['base']},
    corpus=1e7, withdrawal_strategy='fixed',
    withdrawal_params={'monthly_amount': 30000},
    years=30, runs=1000
)

# Deterministic baseline
calc = StaticRetirementCalculator(
    market_return=0.09, inflation_rate=0.052,
    corpus=1e7, withdrawal_strategy='fixed',
    withdrawal_params={'monthly_amount': 30000},
    tax_rate=0.125
)
baseline_results = calc.calculate(years=30)

# Compare
comparison = compare_with_baseline(mc_results, baseline_results)
print_comparison(comparison)
```

---

## Troubleshooting

**Problem**: Simulation runs very slowly
- **Solution**: Reduce `runs` to 500-1000 for testing, use 5000+ only for final analysis

**Problem**: All scenarios show failure
- **Solution**: Withdrawals too high or assumptions too pessimistic. Reduce monthly_amount by 20%

**Problem**: Plots don't appear
- **Solution**: Ensure matplotlib backend is set correctly. Add `import matplotlib; matplotlib.use('TkAgg')` at top

**Problem**: Excel export fails
- **Solution**: Run `pip install openpyxl`, ensure file isn't already open

**Problem**: Import errors
- **Solution**: Run `pip install -r requirements.txt` and verify all dependencies

---

## Further Help

- **Technical Details**: See README.md for complete API reference
- **Examples**: Check examples.py for working code
- **Configuration**: See config_example.ini for all options
- **Issues**: Run test_installation.py to diagnose problems

---

## Disclaimer

This tool provides **planning analysis**, not financial advice. Results depend heavily on input assumptions. Always:
- Test multiple scenarios (conservative, moderate, aggressive)
- Consult qualified financial advisors
- Review and update plans regularly
- Leave safety margins in your planning
- Consider factors beyond simulation (taxes, healthcare, lifestyle changes)

**Version**: 2.0 | **Last Updated**: December 2025
