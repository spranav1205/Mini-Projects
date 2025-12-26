"""
Static Baseline Retirement Simulator

Deterministic retirement calculator using constant returns and inflation.
Useful as a baseline comparison for Monte Carlo simulations.
"""

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows


class StaticRetirementCalculator:
    """
    Deterministic retirement calculator with constant returns and inflation.
    
    Attributes:
        market_return (float): Annual market return rate
        inflation_rate (float): Annual inflation rate
        corpus (float): Initial portfolio value
        withdrawal_strategy (str): 'fixed', 'percentage', or 'dynamic'
        tax_rate (float): Tax rate on withdrawals
    """
    
    def __init__(self, market_return, inflation_rate, corpus, 
                 withdrawal_strategy, withdrawal_params=None, tax_rate=0.125):
        """
        Initialize static calculator.
        
        Args:
            market_return (float): Annual expected return (e.g., 0.09 for 9%)
            inflation_rate (float): Annual expected inflation (e.g., 0.052 for 5.2%)
            corpus (float): Initial portfolio value
            withdrawal_strategy (str): 'fixed', 'percentage', or 'dynamic'
            withdrawal_params (dict): Strategy-specific parameters
            tax_rate (float): Tax rate on withdrawals
        """
        self.market_return = market_return
        self.inflation_rate = inflation_rate
        self.corpus = corpus
        self.withdrawal_strategy = withdrawal_strategy
        self.tax_rate = tax_rate
        self.after_tax_factor = 1 - tax_rate
        self.data = pd.DataFrame()
        
        # Convert annual to monthly
        self.monthly_return = (1 + market_return) ** (1/12) - 1
        self.monthly_inflation = (1 + inflation_rate) ** (1/12) - 1
        
        # Store withdrawal parameters
        withdrawal_params = withdrawal_params or {}
        if withdrawal_strategy == "fixed":
            if 'monthly_amount' not in withdrawal_params:
                raise ValueError("'fixed' strategy requires 'monthly_amount'")
            self.monthly_amount = withdrawal_params['monthly_amount']
            self.annual_lump_sum = withdrawal_params.get('annual_lump_sum', 0)
            self.annual_increment = withdrawal_params.get('annual_increment', 0.0)
            self.grace_period_months = withdrawal_params.get('grace_period_months', 0)
            
        elif withdrawal_strategy == "percentage":
            if 'annual_percentage' not in withdrawal_params:
                raise ValueError("'percentage' strategy requires 'annual_percentage'")
            self.withdrawal_percentage = withdrawal_params['annual_percentage'] / 1200
            
        elif withdrawal_strategy == "dynamic":
            self.min_withdrawal = withdrawal_params.get('min_withdrawal', 0)
            self.max_withdrawal = withdrawal_params.get('max_withdrawal', float('inf'))
            self.target_percentage = withdrawal_params.get('target_percentage', 4.0) / 1200
    
    def calculate_withdrawal(self, month_idx, current_corpus):
        """
        Calculate withdrawal amount based on strategy.
        
        Args:
            month_idx (int): Current month index (0-based)
            current_corpus (float): Current portfolio value
            
        Returns:
            float: Withdrawal amount (pre-tax)
        """
        year = month_idx // 12
        
        if self.withdrawal_strategy == 'fixed':
            if month_idx < self.grace_period_months:
                return 0
            
            # Apply inflation adjustment
            inflation_adjustment = (1 + self.monthly_inflation) ** month_idx
            base_withdrawal = self.monthly_amount * inflation_adjustment
            
            # Apply annual increment
            increment_factor = (1 + self.annual_increment) ** year
            withdrawal = base_withdrawal * increment_factor / self.after_tax_factor
            
            return withdrawal
            
        elif self.withdrawal_strategy == 'percentage':
            return current_corpus * self.withdrawal_percentage
            
        elif self.withdrawal_strategy == 'dynamic':
            base_withdrawal = current_corpus * self.target_percentage
            return max(self.min_withdrawal, min(self.max_withdrawal, base_withdrawal))
        
        return 0
    
    def calculate(self, years, months=0):
        """
        Run deterministic calculation.
        
        Args:
            years (int): Number of years to calculate
            months (int): Additional months beyond full years
            
        Returns:
            pd.DataFrame: Results with columns Year, Month, Corpus
        """
        total_months = years * 12 + months
        
        # Initialize arrays
        corpus_values = np.zeros(total_months)
        corpus_values[0] = self.corpus
        
        for idx in range(1, total_months):
            # Get previous corpus value
            previous_corpus = corpus_values[idx - 1]
            
            # Calculate withdrawal for this month
            withdrawal = self.calculate_withdrawal(idx - 1, previous_corpus)
            
            # Update corpus: apply constant returns and subtract withdrawal
            new_corpus = previous_corpus * (1 + self.monthly_return) - withdrawal
            corpus_values[idx] = new_corpus
            
            # Apply annual lump sum at year end (if using fixed strategy)
            if self.withdrawal_strategy == 'fixed' and self.annual_lump_sum > 0:
                if idx % 12 == 0:
                    year = idx // 12
                    adjusted_lump_sum = self.annual_lump_sum * (1 + self.annual_increment) ** (year - 1)
                    corpus_values[idx] -= adjusted_lump_sum / self.after_tax_factor
        
        # Create DataFrame with results
        months_array = np.arange(total_months)
        self.data = pd.DataFrame({
            'Year': (months_array // 12) + 1,
            'Month': (months_array % 12) + 1,
            'Corpus': corpus_values
        })
        return self.data
    
    def print_summary(self):
        """Print summary of deterministic calculation."""
        if self.data.empty:
            print("No calculation performed yet. Run calculate() first.")
            return
        
        initial = self.data['Corpus'].iloc[0]
        final = self.data['Corpus'].iloc[-1]
        years = len(self.data) / 12
        
        # Calculate inflation-adjusted baseline
        inflation_adjusted = initial * (1 + self.inflation_rate) ** years
        
        print("="*60)
        print("STATIC BASELINE RETIREMENT CALCULATION")
        print("="*60)
        print(f"\nAssumptions:")
        print(f"  Market Return:  {self.market_return*100:.1f}% annually")
        print(f"  Inflation:      {self.inflation_rate*100:.1f}% annually")
        print(f"  Tax Rate:       {self.tax_rate*100:.1f}%")
        
        print(f"\nResults:")
        print(f"  Initial Corpus: ₹{initial/1e7:.2f} crores")
        print(f"  Final Corpus:   ₹{final/1e7:.2f} crores")
        print(f"  Change:         ₹{(final-initial)/1e7:.2f} crores ({((final/initial)-1)*100:.1f}%)")
        
        print(f"\nInflation Analysis:")
        print(f"  Inflation-adjusted baseline: ₹{inflation_adjusted/1e7:.2f} crores")
        if final > inflation_adjusted:
            print(f"  Status: Beats inflation by ₹{(final-inflation_adjusted)/1e7:.2f} crores")
        else:
            print(f"  Status: Falls short by ₹{(inflation_adjusted-final)/1e7:.2f} crores")
        
        # Calculate sustainable withdrawal
        real_return = self.monthly_return - self.monthly_inflation
        sustainable = initial * real_return * self.after_tax_factor * 12
        print(f"\nSustainable Annual Withdrawal:")
        print(f"  Based on real return: ₹{sustainable/1e5:.2f} lakhs/year")
        print(f"  Monthly equivalent:   ₹{sustainable/12:.0f}/month")
        
        print("="*60)
    
    def export_to_excel(self, filename='baseline_results.xlsx'):
        """
        Export baseline calculation results to Excel with multiple sheets.
        
        Args:
            filename (str): Output filename (default: baseline_results.xlsx)
            
        Returns:
            str: Path to created Excel file
        """
        if self.data.empty:
            print("No calculation performed yet. Run calculate() first.")
            return None
        
        wb = Workbook()
        
        # Remove default sheet
        if 'Sheet' in wb.sheetnames:
            wb.remove(wb['Sheet'])
        
        # Sheet 1: Monthly Data
        ws_monthly = wb.create_sheet('Monthly Data', 0)
        
        # Header
        ws_monthly['A1'] = 'Static Baseline - Monthly Breakdown'
        ws_monthly['A1'].font = Font(bold=True, size=14)
        ws_monthly.merge_cells('A1:D1')
        
        # Parameters
        ws_monthly['A3'] = 'Initial Parameters'
        ws_monthly['A3'].font = Font(bold=True)
        ws_monthly['A4'] = 'Initial Corpus:'
        ws_monthly['B4'] = f'₹{self.corpus:,.0f}'
        ws_monthly['A5'] = 'Annual Return:'
        ws_monthly['B5'] = f'{self.market_return:.2%}'
        ws_monthly['A6'] = 'Annual Inflation:'
        ws_monthly['B6'] = f'{self.inflation_rate:.2%}'
        ws_monthly['A7'] = 'Tax Rate:'
        ws_monthly['B7'] = f'{self.tax_rate:.2%}'
        
        # Add withdrawal details based on strategy
        if self.withdrawal_strategy == 'fixed':
            ws_monthly['A8'] = 'Monthly Withdrawal:'
            ws_monthly['B8'] = f'₹{self.monthly_amount:,.0f}'
            if self.annual_lump_sum > 0:
                ws_monthly['A9'] = 'Annual Lump Sum:'
                ws_monthly['B9'] = f'₹{self.annual_lump_sum:,.0f}'
        
        # Monthly data starting from row 11
        for r_idx, row in enumerate(dataframe_to_rows(self.data, index=False, header=True), 11):
            for c_idx, value in enumerate(row, 1):
                cell = ws_monthly.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 11:  # Header row
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
        
        # Adjust column widths
        ws_monthly.column_dimensions['A'].width = 12
        ws_monthly.column_dimensions['B'].width = 18
        ws_monthly.column_dimensions['C'].width = 18
        
        # Sheet 2: Yearly Summary
        ws_yearly = wb.create_sheet('Yearly Summary', 1)
        
        # Aggregate by year
        yearly_data = self.data.groupby('Year').agg({
            'Corpus': ['first', 'last', 'mean']
        }).reset_index()
        yearly_data.columns = ['Year', 'Starting Corpus', 'Ending Corpus', 'Average Corpus']
        yearly_data['Change'] = yearly_data['Ending Corpus'] - yearly_data['Starting Corpus']
        yearly_data['Change %'] = (yearly_data['Change'] / yearly_data['Starting Corpus'] * 100).round(2)
        
        # Write yearly data
        ws_yearly['A1'] = 'Static Baseline - Yearly Summary'
        ws_yearly['A1'].font = Font(bold=True, size=14)
        ws_yearly.merge_cells('A1:F1')
        
        for r_idx, row in enumerate(dataframe_to_rows(yearly_data, index=False, header=True), 3):
            for c_idx, value in enumerate(row, 1):
                cell = ws_yearly.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 3:  # Header row
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color='CCE5FF', end_color='CCE5FF', fill_type='solid')
        
        # Adjust column widths
        for col in ['A', 'B', 'C', 'D', 'E', 'F']:
            ws_yearly.column_dimensions[col].width = 18
        
        # Sheet 3: Summary Statistics
        ws_summary = wb.create_sheet('Summary', 2)
        
        ws_summary['A1'] = 'Static Baseline - Summary Statistics'
        ws_summary['A1'].font = Font(bold=True, size=14)
        ws_summary.merge_cells('A1:B1')
        
        initial_corpus = self.data['Corpus'].iloc[0]
        final_corpus = self.data['Corpus'].iloc[-1]
        months = len(self.data)
        years = months / 12
        
        # Calculate inflation-adjusted baseline
        inflation_factor = (1 + self.inflation_rate) ** years
        inflation_adjusted = initial_corpus * inflation_factor
        
        summary_stats = [
            ('Configuration', ''),
            ('Initial Corpus', f'₹{self.corpus:,.0f}'),
            ('Annual Return', f'{self.market_return:.2%}'),
            ('Annual Inflation', f'{self.inflation_rate:.2%}'),
            ('Tax Rate', f'{self.tax_rate:.2%}'),
            ('Withdrawal Strategy', self.withdrawal_strategy.capitalize()),
            ('', ''),
            ('Simulation Results', ''),
            ('Total Months', months),
            ('Total Years', f'{years:.1f}'),
            ('Final Corpus', f'₹{final_corpus:,.0f}'),
            ('Corpus Change', f'₹{final_corpus - initial_corpus:,.0f}'),
            ('Change %', f'{((final_corpus/initial_corpus - 1)*100):.2f}%'),
            ('', ''),
            ('Inflation Analysis', ''),
            ('Inflation-Adjusted Target', f'₹{inflation_adjusted:,.0f}'),
            ('Beats Inflation?', 'Yes' if final_corpus > inflation_adjusted else 'No'),
            ('Difference', f'₹{final_corpus - inflation_adjusted:,.0f}'),
        ]
        
        for idx, (label, value) in enumerate(summary_stats, 3):
            ws_summary[f'A{idx}'] = label
            ws_summary[f'B{idx}'] = value
            if label and not value:  # Section headers
                ws_summary[f'A{idx}'].font = Font(bold=True, size=12)
            elif label:  # Regular rows
                ws_summary[f'A{idx}'].font = Font(bold=True)
        
        ws_summary.column_dimensions['A'].width = 30
        ws_summary.column_dimensions['B'].width = 25
        
        # Save workbook
        wb.save(filename)
        print(f"\nBaseline results exported to: {filename}")
        return filename


def compare_with_baseline(monte_carlo_results, baseline_result):
    """
    Compare Monte Carlo simulation results with static baseline.
    
    Args:
        monte_carlo_results (dict): Results from simulate_and_measure()
        baseline_result (pd.DataFrame): Results from StaticRetirementCalculator
        
    Returns:
        dict: Comparison metrics
    """
    mc_final_corpuses = monte_carlo_results['final_corpuses']
    baseline_final = baseline_result['Corpus'].iloc[-1]
    
    # Calculate how many MC runs beat the baseline
    beat_baseline = sum(1 for final in mc_final_corpuses if final > baseline_final)
    prob_beat_baseline = beat_baseline / len(mc_final_corpuses)
    
    # Calculate statistics
    mc_mean = np.mean(mc_final_corpuses)
    mc_median = np.median(mc_final_corpuses)
    
    comparison = {
        'baseline_final': baseline_final,
        'mc_mean_final': mc_mean,
        'mc_median_final': mc_median,
        'prob_beat_baseline': prob_beat_baseline,
        'mean_difference': mc_mean - baseline_final,
        'median_difference': mc_median - baseline_final
    }
    
    return comparison


def print_comparison(comparison, scale=1e7, currency_unit="crores"):
    """
    Print comparison between Monte Carlo and baseline results.
    
    Args:
        comparison (dict): Comparison metrics from compare_with_baseline()
        scale (float): Scale for display
        currency_unit (str): Display unit
    """
    print("\n" + "="*60)
    print("MONTE CARLO vs STATIC BASELINE COMPARISON")
    print("="*60)
    
    print(f"\nFinal Corpus:")
    print(f"  Baseline (deterministic): ₹{comparison['baseline_final']/scale:.2f} {currency_unit}")
    print(f"  MC Mean:                  ₹{comparison['mc_mean_final']/scale:.2f} {currency_unit}")
    print(f"  MC Median:                ₹{comparison['mc_median_final']/scale:.2f} {currency_unit}")
    
    print(f"\nDifference from Baseline:")
    print(f"  Mean difference:   ₹{comparison['mean_difference']/scale:.2f} {currency_unit}")
    print(f"  Median difference: ₹{comparison['median_difference']/scale:.2f} {currency_unit}")
    
    print(f"\nProbability Analysis:")
    print(f"  MC scenarios beating baseline: {comparison['prob_beat_baseline']*100:.1f}%")
    
    if comparison['mc_median_final'] > comparison['baseline_final']:
        print(f"\nInterpretation: MC simulations typically perform better than baseline")
    elif comparison['mc_median_final'] < comparison['baseline_final']:
        print(f"\nInterpretation: MC simulations typically underperform baseline")
    else:
        print(f"\nInterpretation: MC simulations align closely with baseline")
    
    print("="*60)


if __name__ == "__main__":
    print("Static Baseline Retirement Calculator")
    print("="*60)
    
    # Example calculation
    calc = StaticRetirementCalculator(
        market_return=0.09,      # 9% annual return
        inflation_rate=0.052,    # 5.2% annual inflation
        corpus=1e7,              # 1 crore
        withdrawal_strategy='fixed',
        withdrawal_params={
            'monthly_amount': 30000,
            'annual_lump_sum': 100000,
            'annual_increment': 0.0,
            'grace_period_months': 0
        },
        tax_rate=0.125
    )
    
    # Run calculation
    results = calc.calculate(years=10, months=0)
    calc.print_summary()
    
    # Export to Excel
    calc.export_to_excel('baseline_example.xlsx')
    
    print("\nNote: Run with Monte Carlo simulation for probabilistic analysis.")
    print("Use compare_with_baseline() to compare MC results with this baseline.")
