"""
Retirement Portfolio Monte Carlo Simulation

This module provides a flexible framework for simulating retirement portfolio
performance using Monte Carlo methods with configurable market returns, inflation,
and withdrawal strategies.
"""

import numpy as np
import pandas as pd

# Parameter presets (annual, nominal). Use for quick scenario setup.
PARAM_PRESETS = {
    'market': {
        'equity': [0.13, 0.20],        # Nifty-like equities
        'optimistic_equity': [0.15, 0.15],
        'balanced_60_40': [0.09, 0.12],
        'conservative_30_70': [0.07, 0.08],
        'gov_bonds': [0.065, 0.05],
        'cash': [0.045, 0.015],
    },
    'inflation': {
        'low': [0.040, 0.012],
        'base': [0.052, 0.020],        # Base case CPI
        'high': [0.06, 0.030],
    }
}

WITHDRAWAL_STRATEGIES = ['fixed', 'percentage', 'dynamic']

class RetirementMCMC:
    """
    Monte Carlo simulator for retirement portfolio planning.
    
    Attributes:
        params (dict): Market and inflation parameters [mean, std]
        corpus (float): Initial portfolio value
        withdrawal_strategy (str): 'fixed', 'percentage', or 'dynamic'
        tax_rate (float): Tax rate on withdrawals (default 0.125)
        data (pd.DataFrame): Simulation results with Year, Month, Corpus columns
    """
    
    def __init__(self, initial_params, corpus, withdrawal_strategy, 
                 withdrawal_params=None, tax_rate=0.125):
        """
        Initialize retirement simulation.
        
        Args:
            initial_params (dict): {'market': [mean, std], 'inflation': [mean, std]}
            corpus (float): Initial portfolio value
            withdrawal_strategy (str): 'fixed', 'percentage', or 'dynamic'
            withdrawal_params (dict): Strategy-specific parameters
            tax_rate (float): Tax rate on withdrawals (0-1 scale)
        """
        self.params = initial_params
        self.corpus = corpus
        self.data = pd.DataFrame()
        self.inflation_rates = []
        self.withdrawal_strategy = withdrawal_strategy
        self.tax_rate = tax_rate
        self.after_tax_factor = 1 - tax_rate

        if withdrawal_strategy not in WITHDRAWAL_STRATEGIES:
            raise ValueError(f"Invalid withdrawal strategy. Choose from {WITHDRAWAL_STRATEGIES}")
        
        # Validate and store withdrawal parameters
        withdrawal_params = withdrawal_params or {}
        if withdrawal_strategy == "fixed":
            if 'monthly_amount' not in withdrawal_params:
                raise ValueError("'fixed' strategy requires 'monthly_amount' in withdrawal_params")
            self.monthly_amount = withdrawal_params['monthly_amount']
            self.annual_lump_sum = withdrawal_params.get('annual_lump_sum', 0)
            self.annual_increment = withdrawal_params.get('annual_increment', 0.0)
            self.grace_period_months = withdrawal_params.get('grace_period_months', 0)
            
        elif withdrawal_strategy == "percentage":
            if 'annual_percentage' not in withdrawal_params:
                raise ValueError("'percentage' strategy requires 'annual_percentage' in withdrawal_params")
            self.withdrawal_percentage = withdrawal_params['annual_percentage'] / 1200  # monthly
            
        elif withdrawal_strategy == "dynamic":
            self.min_withdrawal = withdrawal_params.get('min_withdrawal', 0)
            self.max_withdrawal = withdrawal_params.get('max_withdrawal', float('inf'))
            self.target_percentage = withdrawal_params.get('target_percentage', 4.0) / 1200


    def market_model(self, params, annual=True):
        """
        Generate monthly market return.
        
        Args:
            params (list): [mean, std] for returns
            annual (bool): If True, convert annual params to monthly
            
        Returns:
            float: Monthly market return
        """
        if annual:
            mu = (1 + params[0]) ** (1/12) - 1  # convert annual to monthly via compounding
            sigma = params[1] / np.sqrt(12)
        else:
            mu, sigma = params
        return np.random.normal(mu, sigma)
    
    def inflation_model(self, params, annual=True):
        """
        Generate monthly inflation rate.
        
        Args:
            params (list): [mean, std] for inflation
            annual (bool): If True, convert annual params to monthly
            
        Returns:
            float: Monthly inflation rate
        """
        if annual:
            mu = (1 + params[0]) ** (1/12) - 1  # convert annual to monthly via compounding
            sigma = params[1] / np.sqrt(12)
        else:
            mu, sigma = params
        return np.random.normal(mu, sigma)
    
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
            # Apply grace period
            if month_idx < self.grace_period_months:
                return 0
            
            # Calculate inflation adjustment
            inflation_adjustment = np.prod(1 + np.array(self.inflation_rates))
            base_withdrawal = self.monthly_amount * inflation_adjustment
            
            # Apply annual increment
            increment_factor = (1 + self.annual_increment) ** year
            withdrawal = base_withdrawal * increment_factor / self.after_tax_factor
            
            return withdrawal
            
        elif self.withdrawal_strategy == 'percentage':
            return current_corpus * self.withdrawal_percentage
            
        elif self.withdrawal_strategy == 'dynamic':
            # Dynamic adjusts based on corpus performance
            base_withdrawal = current_corpus * self.target_percentage
            return max(self.min_withdrawal, min(self.max_withdrawal, base_withdrawal))
        
        return 0
    
    def simulate(self, years, months=0):
        """
        Run the retirement simulation.
        
        Args:
            years (int): Number of years to simulate
            months (int): Additional months beyond full years
            
        Returns:
            pd.DataFrame: Results with columns Year, Month, Corpus
        """
        total_months = years * 12 + months

        # Initialize arrays
        corpus_values = np.zeros(total_months)
        corpus_values[0] = self.corpus
        self.inflation_rates = []
        
        for idx in range(1, total_months):
            # Generate monthly market return and inflation
            market_return = self.market_model(self.params['market'])
            inflation_rate = self.inflation_model(self.params['inflation'])
            self.inflation_rates.append(inflation_rate)
            
            # Get previous corpus value
            previous_corpus = corpus_values[idx - 1]
            
            # Calculate withdrawal for this month
            withdrawal = self.calculate_withdrawal(idx - 1, previous_corpus)
            
            # Update corpus: apply returns and subtract withdrawal
            new_corpus = previous_corpus * (1 + market_return) - withdrawal
            corpus_values[idx] = new_corpus
            
            # Apply annual lump sum at year end (if using fixed strategy)
            if self.withdrawal_strategy == 'fixed' and self.annual_lump_sum > 0:
                if idx % 12 == 0:  # End of year
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


def simulate_multiple_runs(initial_params, corpus, withdrawal_strategy, 
                          withdrawal_params, years, months, runs, tax_rate=0.125):
    """
    Run multiple Monte Carlo simulations.
    
    Args:
        initial_params (dict): Market and inflation parameters
        corpus (float): Initial portfolio value
        withdrawal_strategy (str): Withdrawal strategy type
        withdrawal_params (dict): Parameters for withdrawal strategy
        years (int): Years to simulate
        months (int): Additional months
        runs (int): Number of simulation runs
        tax_rate (float): Tax rate on withdrawals
        
    Returns:
        tuple: (traces, final_corpuses) where traces is list of full paths,
               final_corpuses is list of ending values
    """
    final_corpuses = []
    traces = []
    
    for _ in range(runs):
        model = RetirementMCMC(initial_params, corpus, withdrawal_strategy, 
                              withdrawal_params, tax_rate)
        simulation_data = model.simulate(years, months)
        final_corpuses.append(simulation_data['Corpus'].iloc[-1])
        traces.append(simulation_data['Corpus'].values)
    
    return traces, final_corpuses


def simulate_and_measure(initial_params, corpus, withdrawal_strategy, 
                        withdrawal_params, years, months, runs, 
                        confidence=0.90, tax_rate=0.125):
    """
    Run simulations and compute confidence intervals.
    
    Args:
        initial_params (dict): Market and inflation parameters
        corpus (float): Initial portfolio value
        withdrawal_strategy (str): Withdrawal strategy type
        withdrawal_params (dict): Parameters for withdrawal strategy
        years (int): Years to simulate
        months (int): Additional months
        runs (int): Number of simulation runs
        confidence (float): Confidence level (e.g., 0.90 for 90%)
        tax_rate (float): Tax rate on withdrawals
        
    Returns:
        dict: Contains 'traces', 'final_corpuses', 'confidence_interval'
    """
    traces, final_corpuses = simulate_multiple_runs(
        initial_params, corpus, withdrawal_strategy, withdrawal_params, 
        years, months, runs, tax_rate
    )
    
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    ci_lower = np.percentile(final_corpuses, lower_percentile)
    ci_upper = np.percentile(final_corpuses, upper_percentile)
    
    return {
        'traces': traces,
        'final_corpuses': final_corpuses,
        'confidence_interval': (ci_lower, ci_upper),
        'initial_params': initial_params,
        'initial_corpus': corpus
    }



if __name__ == "__main__":
    # Example usage: Conservative retirement planning
    print("Retirement Portfolio Monte Carlo Simulation")
    print("=" * 50)
    
    # Use presets: optimistic equity market + high inflation (conservative assumption)
    initial_params = {
        'market': PARAM_PRESETS['market']['optimistic_equity'],
        'inflation': PARAM_PRESETS['inflation']['high']
    }
    
    corpus = 1e7  # Initial corpus: ₹1 crore
    withdrawal_params = {
        'monthly_amount': 30000,       # ₹30,000 per month
        'annual_lump_sum': 100000,     # ₹1 lakh annual lump sum
        'annual_increment': 0.0,       # No annual increment
        'grace_period_months': 12       # No withdrawals for first 12 months
    }
    
    years, months, runs = 10, 0, 1000
    
    # Run simulation
    print(f"\nSimulating {runs} scenarios over {years} years...")
    results = simulate_and_measure(
        initial_params, corpus, 'fixed', withdrawal_params, 
        years, months, runs, confidence=0.90, tax_rate=0.125
    )

    ci_lo, ci_hi = results['confidence_interval']
    print(f"\n90% Confidence Interval for Final Corpus:")
    print(f"  Lower: ₹{ci_lo/1e7:.2f} crores")
    print(f"  Upper: ₹{ci_hi/1e7:.2f} crores")
    
    # Calculate summary statistics
    final_values = results['final_corpuses']
    print(f"\nFinal Corpus Statistics:")
    print(f"  Mean:   ₹{np.mean(final_values)/1e7:.2f} crores")
    print(f"  Median: ₹{np.median(final_values)/1e7:.2f} crores")
    print(f"  Min:    ₹{np.min(final_values)/1e7:.2f} crores")
    print(f"  Max:    ₹{np.max(final_values)/1e7:.2f} crores")
    
    # Import metrics for visualization (metrics.py will be populated next)
    try:
        from metrics import (
            plot_traces_annual, 
            plot_corpus_histogram,
            probability_beat_inflation,
            print_summary_report,
            export_results_to_excel,
            save_all_plots
        )
        import matplotlib.pyplot as plt
        
        # Print comprehensive report
        print_summary_report(results, withdrawal_params)
        
        # Export to Excel
        export_results_to_excel(results, withdrawal_params, 'simulation_results.xlsx')
        
        # Visualize results
        plot_traces_annual(results['traces'], initial_params['inflation'])
        plt.show()
        plot_corpus_histogram(results['traces'], 5, initial_params['inflation'], corpus)
        plt.show()
        
        # Optional: Save plots to files
        save_all_plots(results, initial_params['inflation'], withdrawal_params, 
                      initial_params['market'], 'plots', 'example_')
        
    except ImportError:
        print("\nNote: Import metrics.py for visualization and additional analysis.")