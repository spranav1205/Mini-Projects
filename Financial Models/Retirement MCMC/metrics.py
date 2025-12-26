"""
Metrics and Visualization for Retirement Portfolio Simulation

This module provides comprehensive metrics, analysis functions, and visualization
tools for evaluating retirement portfolio simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows


def annualize_traces(traces: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Convert monthly traces to annual (end-of-year) values.
    
    Args:
        traces: List of monthly corpus arrays from simulations
        
    Returns:
        tuple: (annual_traces, n_years) where annual_traces has shape (runs, n_years)
    """
    if not traces:
        return np.array([]), 0
    
    total_months = len(traces[0])
    n_years = total_months // 12
    year_end_idx = (np.arange(1, n_years + 1) * 12) - 1
    annual = np.array([np.asarray(t)[year_end_idx] for t in traces])
    
    return annual, n_years


def probability_beat_inflation(initial_corpus: float, final_corpuses: List[float], 
                               inflation_rate: List[float], years: int) -> Tuple[float, float]:
    """
    Calculate probability that final corpus exceeds inflation-adjusted initial corpus.
    
    Args:
        initial_corpus: Starting corpus amount
        final_corpuses: List of final corpus values from simulations
        inflation_rate: Annual inflation parameters [mean, std]
        years: Number of years simulated
    
    Returns:
        tuple: (probability, inflation_adjusted_baseline)
    """
    inflation_adjusted = initial_corpus * (1 + inflation_rate[0]) ** years
    beat_inflation_count = sum(1 for final in final_corpuses if final > inflation_adjusted)
    probability = beat_inflation_count / len(final_corpuses)
    
    return probability, inflation_adjusted


def calculate_ruin_probability(final_corpuses: List[float], threshold: float = 0) -> float:
    """
    Calculate probability of portfolio ruin (corpus falling below threshold).
    
    Args:
        final_corpuses: List of final corpus values
        threshold: Minimum acceptable corpus value
        
    Returns:
        float: Probability of ruin (0-1)
    """
    ruin_count = sum(1 for final in final_corpuses if final <= threshold)
    return ruin_count / len(final_corpuses)


def calculate_shortfall_risk(final_corpuses: List[float], target: float) -> Tuple[float, float]:
    """
    Calculate shortfall risk: probability and average magnitude of falling short of target.
    
    Args:
        final_corpuses: List of final corpus values
        target: Target corpus value
        
    Returns:
        tuple: (shortfall_probability, average_shortfall_amount)
    """
    shortfalls = [target - final for final in final_corpuses if final < target]
    shortfall_prob = len(shortfalls) / len(final_corpuses)
    avg_shortfall = np.mean(shortfalls) if shortfalls else 0
    
    return shortfall_prob, avg_shortfall


def calculate_sustainability_metrics(traces: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate portfolio sustainability metrics across all simulations.
    
    Args:
        traces: List of corpus traces from simulations
        
    Returns:
        dict: Various sustainability metrics
    """
    traces_array = np.array(traces)
    
    # Years until corpus becomes negative (if at all)
    years_to_depletion = []
    for trace in traces:
        negative_idx = np.where(trace <= 0)[0]
        if len(negative_idx) > 0:
            years_to_depletion.append(negative_idx[0] / 12)
    
    metrics = {
        'depletion_probability': len(years_to_depletion) / len(traces),
        'avg_years_to_depletion': np.mean(years_to_depletion) if years_to_depletion else None,
        'median_years_to_depletion': np.median(years_to_depletion) if years_to_depletion else None,
        'min_corpus_mean': np.mean([np.min(trace) for trace in traces]),
        'min_corpus_median': np.median([np.min(trace) for trace in traces]),
    }
    
    return metrics


def calculate_withdrawal_sustainability(traces: List[np.ndarray], market_params: List[float],
                                        inflation_params: List[float], 
                                        tax_rate: float = 0.125) -> np.ndarray:
    """
    Calculate theoretical maximum sustainable withdrawal at each time point.
    
    Args:
        traces: List of corpus traces
        market_params: [mean, std] for market returns (annual)
        inflation_params: [mean, std] for inflation (annual)
        tax_rate: Tax rate on withdrawals
        
    Returns:
        np.ndarray: Mean sustainable withdrawals over time
    """
    # Convert to monthly
    mean_return = (1 + market_params[0]) ** (1/12) - 1
    mean_inflation = (1 + inflation_params[0]) ** (1/12) - 1
    after_tax_factor = 1 - tax_rate
    
    traces_array = np.array(traces)
    months = traces_array.shape[1]
    
    # Max withdrawal = corpus * (real_return) * after_tax_factor
    sustainable = np.zeros((len(traces), months))
    for month_idx in range(months):
        corpus_at_month = traces_array[:, month_idx]
        sustainable[:, month_idx] = corpus_at_month * (mean_return - mean_inflation) * after_tax_factor
    
    return np.mean(sustainable, axis=0)


def print_summary_report(results: Dict, withdrawal_params: Dict, 
                        currency_unit: str = "crores", scale: float = 1e7):
    """
    Print comprehensive summary report of simulation results.
    
    Args:
        results: Dictionary containing simulation results
        withdrawal_params: Withdrawal strategy parameters
        currency_unit: Display unit for currency
        scale: Conversion scale for display
    """
    print("\n" + "="*70)
    print("RETIREMENT PORTFOLIO SIMULATION - SUMMARY REPORT")
    print("="*70)
    
    final_corpuses = results['final_corpuses']
    initial_corpus = results['initial_corpus']
    initial_params = results['initial_params']
    
    # Basic statistics
    print(f"\nInitial Corpus: ₹{initial_corpus/scale:.2f} {currency_unit}")
    print(f"\nFinal Corpus Statistics:")
    print(f"  Mean:       ₹{np.mean(final_corpuses)/scale:.2f} {currency_unit}")
    print(f"  Median:     ₹{np.median(final_corpuses)/scale:.2f} {currency_unit}")
    print(f"  Std Dev:    ₹{np.std(final_corpuses)/scale:.2f} {currency_unit}")
    print(f"  Min:        ₹{np.min(final_corpuses)/scale:.2f} {currency_unit}")
    print(f"  Max:        ₹{np.max(final_corpuses)/scale:.2f} {currency_unit}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(final_corpuses, p)
        print(f"  {p}th:       ₹{val/scale:.2f} {currency_unit}")
    
    # Confidence interval
    ci_lo, ci_hi = results['confidence_interval']
    print(f"\n90% Confidence Interval: ₹{ci_lo/scale:.2f} - ₹{ci_hi/scale:.2f} {currency_unit}")
    
    # Risk metrics
    years = len(results['traces'][0]) // 12
    prob, inflation_baseline = probability_beat_inflation(
        initial_corpus, final_corpuses, initial_params['inflation'], years
    )
    print(f"\nRisk Metrics:")
    print(f"  Probability of beating inflation: {prob*100:.1f}%")
    print(f"  Inflation-adjusted baseline: ₹{inflation_baseline/scale:.2f} {currency_unit}")
    
    ruin_prob = calculate_ruin_probability(final_corpuses)
    print(f"  Probability of ruin (corpus ≤ 0): {ruin_prob*100:.1f}%")
    
    shortfall_prob, avg_shortfall = calculate_shortfall_risk(final_corpuses, inflation_baseline)
    print(f"  Probability of shortfall vs inflation: {shortfall_prob*100:.1f}%")
    if avg_shortfall > 0:
        print(f"  Average shortfall amount: ₹{avg_shortfall/scale:.2f} {currency_unit}")
    
    # Sustainability
    sustainability = calculate_sustainability_metrics(results['traces'])
    print(f"\nSustainability Metrics:")
    print(f"  Depletion probability: {sustainability['depletion_probability']*100:.1f}%")
    if sustainability['avg_years_to_depletion']:
        print(f"  Average years to depletion: {sustainability['avg_years_to_depletion']:.1f}")
    
    # Withdrawal info
    print(f"\nWithdrawal Parameters:")
    for key, value in withdrawal_params.items():
        if isinstance(value, (int, float)):
            if 'amount' in key.lower():
                print(f"  {key}: ₹{value:,.0f}")
            elif 'percentage' in key.lower():
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("="*70 + "\n")


def plot_traces_annual(traces: List[np.ndarray], inflation_params: List[float], 
                       max_traces: int = 25, currency_unit: str = "crores", 
                       scale: float = 1e7, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot multiple simulation traces at annual frequency.
    
    Args:
        traces: List of corpus traces from simulations
        inflation_params: [mean, std] for inflation
        max_traces: Maximum number of individual traces to show
        currency_unit: Display unit for currency
        scale: Conversion scale for display
        figsize: Figure size (width, height)
    """
    annual, n_years = annualize_traces(traces)
    if n_years == 0:
        print("No data to plot")
        return

    x = np.arange(1, n_years + 1)
    initial_corpus = float(traces[0][0]) / scale
    baseline = [initial_corpus * (1 + inflation_params[0]) ** year for year in range(n_years)]

    plt.figure(figsize=figsize)
    
    # Sample traces
    n_show = min(len(annual), max_traces)
    for y in annual[:n_show]:
        plt.plot(x, y / scale, color='tab:blue', alpha=0.25, linewidth=1)

    # Statistics
    median = np.median(annual, axis=0) / scale
    p10, p90 = np.percentile(annual, [10, 90], axis=0) / scale
    
    plt.plot(x, median, color='tab:blue', linewidth=2, label='Median', zorder=5)
    plt.plot(x, baseline, color='tab:orange', linestyle='--', linewidth=2, 
             label='Inflation Baseline', zorder=5)
    plt.fill_between(x, p10, p90, color='tab:blue', alpha=0.15, label='10-90% Range')

    plt.xlabel('Year', fontsize=11)
    plt.ylabel(f'Corpus ({currency_unit})', fontsize=11)
    plt.title('Retirement Corpus Simulation - Annual View', fontsize=13, fontweight='bold')
    plt.xticks(x, [str(i) for i in x])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()


def plot_corpus_histogram(traces: List[np.ndarray], year: int, 
                          inflation_params: List[float], initial_corpus: float,
                          bins: int = 50, currency_unit: str = "crores", 
                          scale: float = 1e7, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot histogram of corpus values at a specific year.
    
    Args:
        traces: List of corpus traces from simulations
        year: Which year to analyze (1-indexed)
        inflation_params: [mean, std] for inflation
        initial_corpus: Starting corpus amount
        bins: Number of histogram bins
        currency_unit: Display unit for currency
        scale: Conversion scale for display
        figsize: Figure size
    """
    annual, n_years = annualize_traces(traces)
    
    if year < 1 or year > n_years:
        print(f"Year {year} is out of range. Valid range: 1-{n_years}")
        return
    
    corpus_at_year = annual[:, year - 1]
    inflation_adjusted = initial_corpus * (1 + inflation_params[0]) ** year
    
    plt.figure(figsize=figsize)
    
    # Histogram
    plt.hist(corpus_at_year / scale, bins=bins, color='tab:blue', 
             alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Statistics lines
    median = np.median(corpus_at_year) / scale
    mean = np.mean(corpus_at_year) / scale
    p10, p90 = np.percentile(corpus_at_year, [10, 90]) / scale
    inflation_line = inflation_adjusted / scale
    
    plt.axvline(median, color='tab:blue', linestyle='-', linewidth=2, 
                label=f'Median: ₹{median:.2f}')
    plt.axvline(mean, color='tab:green', linestyle='--', linewidth=2, 
                label=f'Mean: ₹{mean:.2f}')
    plt.axvline(inflation_line, color='tab:orange', linestyle='--', linewidth=2, 
                label=f'Inflation: ₹{inflation_line:.2f}')
    plt.axvline(p10, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7, 
                label=f'10th %ile: ₹{p10:.2f}')
    plt.axvline(p90, color='tab:red', linestyle=':', linewidth=1.5, alpha=0.7, 
                label=f'90th %ile: ₹{p90:.2f}')
    
    plt.xlabel(f'Corpus ({currency_unit})', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'Corpus Distribution at Year {year}', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Print statistics
    beat_inflation = sum(1 for c in corpus_at_year if c > inflation_adjusted) / len(corpus_at_year)
    print(f"\nYear {year} Statistics:")
    print(f"  Probability of beating inflation: {beat_inflation*100:.1f}%")
    print(f"  Mean corpus: ₹{mean:.2f} {currency_unit}")
    print(f"  Median corpus: ₹{median:.2f} {currency_unit}")


def plot_withdrawal_analysis(traces: List[np.ndarray], market_params: List[float],
                            inflation_params: List[float], withdrawal_params: Dict,
                            max_traces: int = 20, currency_unit: str = "lakhs",
                            scale: float = 1e5, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot theoretical maximum sustainable withdrawal vs requested withdrawal.
    
    Args:
        traces: List of corpus traces from simulations
        market_params: [mean, std] for market returns
        inflation_params: [mean, std] for inflation
        withdrawal_params: Dictionary with withdrawal parameters
        max_traces: Maximum traces to show
        currency_unit: Display unit for currency
        scale: Conversion scale for display
        figsize: Figure size
    """
    tax_rate = withdrawal_params.get('tax_rate', 0.125)
    sustainable = calculate_withdrawal_sustainability(traces, market_params, 
                                                     inflation_params, tax_rate)
    
    months = len(traces[0])
    x = np.arange(1, months + 1)
    
    # Calculate requested withdrawal over time
    monthly_amount = withdrawal_params.get('monthly_amount', 0)
    annual_lump_sum = withdrawal_params.get('annual_lump_sum', 0)
    annual_increment = withdrawal_params.get('annual_increment', 0.0)
    
    mean_inflation = (1 + inflation_params[0]) ** (1/12) - 1
    after_tax = 1 - tax_rate
    
    monthly_request = (monthly_amount + annual_lump_sum / 12) / after_tax
    requested = [monthly_request * (1 + mean_inflation) ** month * 
                (1 + annual_increment) ** (month // 12) for month in range(months)]
    
    # Plot theoretical sustainable withdrawals
    traces_array = np.array(traces)
    mean_return = (1 + market_params[0]) ** (1/12) - 1
    
    theoretical_withdrawals = np.zeros((len(traces), months))
    for month_idx in range(months):
        corpus_at_month = traces_array[:, month_idx]
        theoretical_withdrawals[:, month_idx] = (corpus_at_month * 
                                                 (mean_return - mean_inflation) * after_tax)
    
    plt.figure(figsize=figsize)
    
    # Sample traces
    n_show = min(len(theoretical_withdrawals), max_traces)
    for k in theoretical_withdrawals[:n_show]:
        plt.plot(x, k / scale, color='tab:green', alpha=0.25, linewidth=1)
    
    # Mean and confidence intervals
    mean_withdrawals = np.mean(theoretical_withdrawals, axis=0)
    p10, p90 = np.percentile(theoretical_withdrawals, [10, 90], axis=0)
    
    plt.plot(x, mean_withdrawals / scale, color='tab:green', linewidth=2, 
             label='Mean Sustainable Withdrawal', zorder=5)
    plt.fill_between(x, p10 / scale, p90 / scale, color='tab:green', 
                     alpha=0.15, label='10-90% Range')
    plt.plot(x, np.array(requested) / scale, color='tab:red', linewidth=2, 
             label='Requested Withdrawal', zorder=5)
    
    plt.xlabel('Month', fontsize=11)
    plt.ylabel(f'Withdrawal Amount ({currency_unit})', fontsize=11)
    plt.title('Sustainable vs Requested Withdrawal', fontsize=13, fontweight='bold')
    plt.xticks(x[::12], [str(int(i / 12)) for i in x[::12]])
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()


def plot_risk_heatmap(param_ranges: Dict, corpus: float, withdrawal_params: Dict,
                     years: int, runs: int = 500, figsize: Tuple[int, int] = (10, 8)):
    """
    Create a heatmap showing success probability across different parameter combinations.
    
    Args:
        param_ranges: Dictionary with ranges for market/inflation parameters
        corpus: Initial corpus
        withdrawal_params: Withdrawal parameters
        years: Simulation years
        runs: Number of runs per combination
        figsize: Figure size
    """
    from simulation import simulate_and_measure, PARAM_PRESETS
    
    market_scenarios = param_ranges.get('market', list(PARAM_PRESETS['market'].values())[:3])
    inflation_scenarios = param_ranges.get('inflation', list(PARAM_PRESETS['inflation'].values()))
    
    success_matrix = np.zeros((len(inflation_scenarios), len(market_scenarios)))
    
    for i, inflation in enumerate(inflation_scenarios):
        for j, market in enumerate(market_scenarios):
            params = {'market': market, 'inflation': inflation}
            results = simulate_and_measure(params, corpus, 'fixed', 
                                          withdrawal_params, years, 0, runs)
            
            # Success = corpus > inflation-adjusted initial
            prob, _ = probability_beat_inflation(corpus, results['final_corpuses'], 
                                                inflation, years)
            success_matrix[i, j] = prob * 100
    
    plt.figure(figsize=figsize)
    im = plt.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    plt.colorbar(im, label='Success Probability (%)')
    plt.xlabel('Market Scenario', fontsize=11)
    plt.ylabel('Inflation Scenario', fontsize=11)
    plt.title('Retirement Plan Success Rate Heatmap', fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(len(inflation_scenarios)):
        for j in range(len(market_scenarios)):
            text = plt.text(j, i, f'{success_matrix[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.xticks(range(len(market_scenarios)), 
               [f"M{i+1}" for i in range(len(market_scenarios))])
    plt.yticks(range(len(inflation_scenarios)), 
               [f"I{i+1}" for i in range(len(inflation_scenarios))])
    plt.tight_layout()


def export_results_to_csv(results: Dict, filename: str = "simulation_results.csv"):
    """
    Export simulation results to CSV file.
    
    Args:
        results: Dictionary containing simulation results
        filename: Output filename
    """
    final_corpuses = results['final_corpuses']
    
    df = pd.DataFrame({
        'simulation_run': range(1, len(final_corpuses) + 1),
        'final_corpus': final_corpuses
    })
    
    # Add summary statistics
    summary = {
        'mean': np.mean(final_corpuses),
        'median': np.median(final_corpuses),
        'std': np.std(final_corpuses),
        'min': np.min(final_corpuses),
        'max': np.max(final_corpuses),
        'p10': np.percentile(final_corpuses, 10),
        'p25': np.percentile(final_corpuses, 25),
        'p75': np.percentile(final_corpuses, 75),
        'p90': np.percentile(final_corpuses, 90),
    }
    
    df.to_csv(filename, index=False)
    
    # Save summary separately
    summary_filename = filename.replace('.csv', '_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_filename, index=False)
    
    print(f"Results exported to {filename}")
    print(f"Summary exported to {summary_filename}")


def export_results_to_excel(results: Dict, withdrawal_params: Dict, filename: str = 'simulation_results.xlsx'):
    """
    Export Monte Carlo simulation results to Excel with multiple sheets.
    
    Args:
        results: Dictionary from simulate_and_measure()
        withdrawal_params: Withdrawal parameters dict
        filename: Output filename (default: simulation_results.xlsx)
    
    Returns:
        Path to created Excel file
    """
    wb = Workbook()
    
    # Remove default sheet
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])
    
    # Sheet 1: Summary Statistics
    ws_summary = wb.create_sheet('Summary', 0)
    
    ws_summary['A1'] = 'Monte Carlo Simulation - Summary Statistics'
    ws_summary['A1'].font = Font(bold=True, size=14)
    ws_summary.merge_cells('A1:B1')
    
    final_corpuses = results['final_corpuses']
    traces = results['traces']
    
    # Calculate key metrics
    success_prob, _ = probability_beat_inflation(
        results['initial_corpus'], final_corpuses, 
        results['initial_params']['inflation'], len(traces[0]) // 12
    )
    ruin_prob = calculate_ruin_probability(final_corpuses)
    shortfall_prob, avg_shortfall = calculate_shortfall_risk(
        final_corpuses, results['initial_corpus'] * 
        (1 + results['initial_params']['inflation'][0]) ** (len(traces[0]) // 12)
    )
    
    summary_stats = [
        ('Simulation Configuration', ''),
        ('Number of Runs', len(final_corpuses)),
        ('Time Horizon (months)', len(traces[0])),
        ('Time Horizon (years)', f'{len(traces[0])/12:.1f}'),
        ('', ''),
        ('Final Corpus Statistics', ''),
        ('Mean', f'₹{np.mean(final_corpuses):,.0f}'),
        ('Median', f'₹{np.median(final_corpuses):,.0f}'),
        ('Std Dev', f'₹{np.std(final_corpuses):,.0f}'),
        ('Minimum', f'₹{np.min(final_corpuses):,.0f}'),
        ('Maximum', f'₹{np.max(final_corpuses):,.0f}'),
        ('10th Percentile', f'₹{np.percentile(final_corpuses, 10):,.0f}'),
        ('25th Percentile', f'₹{np.percentile(final_corpuses, 25):,.0f}'),
        ('75th Percentile', f'₹{np.percentile(final_corpuses, 75):,.0f}'),
        ('90th Percentile', f'₹{np.percentile(final_corpuses, 90):,.0f}'),
        ('', ''),
        ('Risk Metrics', ''),
        ('Success Probability', f'{success_prob:.1f}%'),
        ('Ruin Probability', f'{ruin_prob:.1f}%'),
        ('Shortfall Probability', f'{shortfall_prob:.1f}%'),
        ('Avg Shortfall (when occurs)', f'₹{avg_shortfall:,.0f}'),
    ]
    
    # Add withdrawal parameters
    if 'monthly_amount' in withdrawal_params:
        summary_stats.extend([
            ('', ''),
            ('Withdrawal Parameters', ''),
            ('Monthly Amount', f'₹{withdrawal_params["monthly_amount"]:,.0f}'),
        ])
        if 'annual_lump_sum' in withdrawal_params and withdrawal_params['annual_lump_sum'] > 0:
            summary_stats.append(('Annual Lump Sum', f'₹{withdrawal_params["annual_lump_sum"]:,.0f}'))
    
    for idx, (label, value) in enumerate(summary_stats, 3):
        ws_summary[f'A{idx}'] = label
        ws_summary[f'B{idx}'] = value
        if label and not value:  # Section headers
            ws_summary[f'A{idx}'].font = Font(bold=True, size=12)
        elif label:  # Regular rows
            ws_summary[f'A{idx}'].font = Font(bold=True)
    
    ws_summary.column_dimensions['A'].width = 30
    ws_summary.column_dimensions['B'].width = 25
    
    # Sheet 2: Percentile Traces
    ws_percentiles = wb.create_sheet('Percentile Traces', 1)
    
    # Calculate percentile traces
    traces_array = np.array(traces)
    months = len(traces[0])
    
    percentile_data = {
        'Month': list(range(1, months + 1)),
        'Year': [(m-1)//12 + 1 for m in range(1, months + 1)],
        'P10': [np.percentile(traces_array[:, m], 10) for m in range(months)],
        'P25': [np.percentile(traces_array[:, m], 25) for m in range(months)],
        'Median': [np.percentile(traces_array[:, m], 50) for m in range(months)],
        'P75': [np.percentile(traces_array[:, m], 75) for m in range(months)],
        'P90': [np.percentile(traces_array[:, m], 90) for m in range(months)],
        'Mean': [np.mean(traces_array[:, m]) for m in range(months)]
    }
    df_percentiles = pd.DataFrame(percentile_data)
    
    ws_percentiles['A1'] = 'Monte Carlo - Corpus Evolution Percentiles'
    ws_percentiles['A1'].font = Font(bold=True, size=14)
    ws_percentiles.merge_cells('A1:H1')
    
    for r_idx, row in enumerate(dataframe_to_rows(df_percentiles, index=False, header=True), 3):
        for c_idx, value in enumerate(row, 1):
            cell = ws_percentiles.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 3:  # Header row
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
    
    # Adjust column widths
    for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        ws_percentiles.column_dimensions[col].width = 15
    
    # Sheet 3: Final Corpus Distribution
    ws_final = wb.create_sheet('Final Corpus', 2)
    
    final_data = pd.DataFrame({
        'Run': list(range(1, len(final_corpuses) + 1)),
        'Final Corpus': final_corpuses
    })
    final_data = final_data.sort_values('Final Corpus', ascending=False).reset_index(drop=True)
    
    ws_final['A1'] = 'Monte Carlo - Final Corpus by Run'
    ws_final['A1'].font = Font(bold=True, size=14)
    ws_final.merge_cells('A1:B1')
    
    for r_idx, row in enumerate(dataframe_to_rows(final_data, index=False, header=True), 3):
        for c_idx, value in enumerate(row, 1):
            cell = ws_final.cell(row=r_idx, column=c_idx, value=value)
            if r_idx == 3:  # Header row
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
    
    ws_final.column_dimensions['A'].width = 12
    ws_final.column_dimensions['B'].width = 20
    
    # Save workbook
    wb.save(filename)
    print(f"\nMonte Carlo results exported to: {filename}")
    return filename


def save_all_plots(results: Dict, inflation_params: List[float], withdrawal_params: Dict, 
                   market_params: List[float] = None, output_dir: str = 'plots', prefix: str = ''):
    """
    Generate and save all visualization plots to files.
    
    Args:
        results: Dictionary from simulate_and_measure()
        inflation_params: Inflation parameters [mean, std]
        withdrawal_params: Withdrawal parameters dict
        market_params: Market parameters [mean, std] (optional, from results if not provided)
        output_dir: Directory to save plots (default: 'plots')
        prefix: Filename prefix (default: '')
    
    Returns:
        List of saved filenames
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get market params from results if not provided
    if market_params is None:
        market_params = results['initial_params']['market']
    
    saved_files = []
    
    # Plot 1: Annual Traces
    plt.figure(figsize=(14, 7))
    plot_traces_annual(results['traces'], inflation_params)
    filename = os.path.join(output_dir, f'{prefix}traces_annual.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filename)
    print(f"Saved: {filename}")
    
    # Plot 2: Corpus Histogram - plot final corpus distribution
    plt.figure(figsize=(10, 6))
    final_corpuses = results['final_corpuses']
    plt.hist(np.array(final_corpuses) / 1e7, bins=50, color='tab:blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Final Corpus (crores)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Final Corpus Distribution', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{prefix}corpus_histogram.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filename)
    print(f"Saved: {filename}")
    
    # Plot 3: Withdrawal Analysis
    plt.figure(figsize=(12, 6))
    # Add tax_rate to withdrawal_params if not present
    if 'tax_rate' not in withdrawal_params:
        withdrawal_params_with_tax = withdrawal_params.copy()
        withdrawal_params_with_tax['tax_rate'] = 0.125
    else:
        withdrawal_params_with_tax = withdrawal_params
    
    plot_withdrawal_analysis(
        results['traces'],
        market_params,
        inflation_params,
        withdrawal_params_with_tax
    )
    filename = os.path.join(output_dir, f'{prefix}withdrawal_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(filename)
    print(f"Saved: {filename}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    return saved_files

