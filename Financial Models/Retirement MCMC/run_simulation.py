"""
Run retirement simulation from configuration file.

Usage:
    python run_simulation.py config_example.ini
    python run_simulation.py my_config.ini
"""

import sys
import os
import configparser
import matplotlib.pyplot as plt
from simulation import simulate_and_measure, PARAM_PRESETS
from metrics import (
    print_summary_report,
    plot_traces_annual,
    plot_corpus_histogram,
    plot_withdrawal_analysis,
    export_results_to_csv
)


def load_config(config_file):
    """Load and parse configuration file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def parse_market_params(config):
    """Parse market parameters from config."""
    if 'market_preset' in config['portfolio']:
        preset = config['portfolio']['market_preset']
        if preset in PARAM_PRESETS['market']:
            return PARAM_PRESETS['market'][preset]
        else:
            print(f"Warning: Unknown market preset '{preset}', using balanced_60_40")
            return PARAM_PRESETS['market']['balanced_60_40']
    else:
        return [
            float(config['portfolio']['market_return']),
            float(config['portfolio']['market_volatility'])
        ]


def parse_inflation_params(config):
    """Parse inflation parameters from config."""
    if 'inflation_preset' in config['portfolio']:
        preset = config['portfolio']['inflation_preset']
        if preset in PARAM_PRESETS['inflation']:
            return PARAM_PRESETS['inflation'][preset]
        else:
            print(f"Warning: Unknown inflation preset '{preset}', using base")
            return PARAM_PRESETS['inflation']['base']
    else:
        return [
            float(config['portfolio']['inflation_rate']),
            float(config['portfolio']['inflation_volatility'])
        ]


def parse_withdrawal_params(config):
    """Parse withdrawal strategy parameters from config."""
    strategy = config['withdrawal']['strategy'].lower()
    params = {}
    
    if strategy == 'fixed':
        params['monthly_amount'] = float(config['withdrawal'].get('monthly_amount', 0))
        params['annual_lump_sum'] = float(config['withdrawal'].get('annual_lump_sum', 0))
        params['annual_increment'] = float(config['withdrawal'].get('annual_increment', 0.0))
        params['grace_period_months'] = int(config['withdrawal'].get('grace_period_months', 0))
    
    elif strategy == 'percentage':
        params['annual_percentage'] = float(config['withdrawal'].get('annual_percentage', 4.0))
    
    elif strategy == 'dynamic':
        params['min_withdrawal'] = float(config['withdrawal'].get('min_withdrawal', 0))
        params['max_withdrawal'] = float(config['withdrawal'].get('max_withdrawal', float('inf')))
        params['target_percentage'] = float(config['withdrawal'].get('target_percentage', 4.0))
    
    return strategy, params


def run_from_config(config_file):
    """Run simulation based on configuration file."""
    print("="*70)
    print("RETIREMENT PORTFOLIO SIMULATION FROM CONFIG")
    print("="*70)
    
    # Load configuration
    config = load_config(config_file)
    scenario_name = config['scenario'].get('name', 'Unnamed')
    print(f"\nLoading configuration from: {config_file}")
    print(f"Scenario: {scenario_name}")
    print(f"Description: {config['scenario'].get('description', 'No description')}")
    
    # Parse parameters
    initial_params = {
        'market': parse_market_params(config),
        'inflation': parse_inflation_params(config)
    }
    
    corpus = float(config['portfolio']['initial_corpus'])
    tax_rate = float(config['portfolio'].get('tax_rate', 0.125))
    
    withdrawal_strategy, withdrawal_params = parse_withdrawal_params(config)
    
    years = int(config['simulation']['years'])
    months = int(config['simulation'].get('months', 0))
    runs = int(config['simulation']['runs'])
    confidence = float(config['simulation'].get('confidence', 0.90))
    
    # Display configuration
    print(f"\nPortfolio Configuration:")
    print(f"  Initial Corpus: ₹{corpus/1e7:.2f} crores")
    print(f"  Market: {initial_params['market']}")
    print(f"  Inflation: {initial_params['inflation']}")
    print(f"  Tax Rate: {tax_rate*100}%")
    
    print(f"\nWithdrawal Strategy: {withdrawal_strategy}")
    for key, value in withdrawal_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nSimulation Parameters:")
    print(f"  Duration: {years} years, {months} months")
    print(f"  Monte Carlo Runs: {runs}")
    print(f"  Confidence Level: {confidence*100}%")
    
    # Run simulation
    print(f"\nRunning simulation...")
    results = simulate_and_measure(
        initial_params,
        corpus,
        withdrawal_strategy,
        withdrawal_params,
        years,
        months,
        runs,
        confidence,
        tax_rate
    )
    
    print("Simulation complete!")
    
    # Print summary report
    print_summary_report(results, withdrawal_params)
    
    # Plot saving options
    save_plots = config['output'].getboolean('save_plots', True)
    plots_dir = config['output'].get('plots_dir', 'plots')
    plots_prefix = config['output'].get('plots_prefix', '').strip()
    if not plots_prefix:
        safe_name = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in scenario_name.strip())
        plots_prefix = f"{safe_name}_" if safe_name else ""
    if save_plots:
        os.makedirs(plots_dir, exist_ok=True)

    def save_plot(filename: str) -> None:
        if not save_plots:
            return
        path = os.path.join(plots_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {path}")

    # Generate visualizations based on config
    if config['output'].getboolean('plot_traces', True):
        print("\nGenerating annual traces plot...")
        plot_traces_annual(results['traces'], initial_params['inflation'])
        save_plot(f"{plots_prefix}traces_annual.png")
        plt.show()
    
    if config['output'].getboolean('plot_histogram', True):
        histogram_year = int(config['output'].get('histogram_year', 5))
        print(f"\nGenerating corpus histogram for year {histogram_year}...")
        plot_corpus_histogram(
            results['traces'],
            histogram_year,
            initial_params['inflation'],
            corpus
        )
        save_plot(f"{plots_prefix}corpus_histogram_year_{histogram_year}.png")
        plt.show()
        
    
    if config['output'].getboolean('plot_withdrawal', True):
        print("\nGenerating withdrawal analysis plot...")
        withdrawal_params['tax_rate'] = tax_rate
        plot_withdrawal_analysis(
            results['traces'],
            initial_params['market'],
            initial_params['inflation'],
            withdrawal_params
        )
        save_plot(f"{plots_prefix}withdrawal_analysis.png")
        plt.show()
    
    # Export to CSV if requested
    if config['output'].getboolean('export_csv', False):
        csv_filename = config['output'].get('csv_filename', 'retirement_results.csv')
        print(f"\nExporting results to {csv_filename}...")
        export_results_to_csv(results, csv_filename)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run_simulation.py <config_file>")
        print("\nExample:")
        print("  python run_simulation.py config_example.ini")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        run_from_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
