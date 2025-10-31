import subprocess
import pandas as pd
import numpy as np
import os

# List of seeds to try
SEEDS = [1,2,3,4,5,6,7,8,9,10]

# Use only the six drugs we care about (exclude Tegafur to avoid missing true value)
DRUGS = [
    'Ifosfamide',
    'Pyrazinamide (Pyrazinoic acid amide)',
    'Zidovudine (Retrovir)',
    'Metformin hydrochloride (Glucophage)',
    'Captopril (Capoten)',
    'Tiopronin (Thiola)'
]

required_drugs_arg = ','.join(DRUGS)

# Ground-truth values for the selected drugs (used for summary/error metrics)
true_values = {
    'Ifosfamide': 0.566,
    'Pyrazinamide (Pyrazinoic acid amide)': 0.39,
    'Zidovudine (Retrovir)': 0.36,
    'Metformin hydrochloride (Glucophage)': 0.94,
    'Captopril (Capoten)': 0.61,
    'Tiopronin (Thiola)': 0.38
}

def normalize_name(s):
    if pd.isna(s):
        return ''
    return str(s).lower().replace('"', '').replace("'", '').replace(' ', '').strip()

# Output directory for per-seed CSVs
OUTPUT_DIR = 'seed_predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output CSVs for each run
output_files = []

for seed in SEEDS:
    output_csv = os.path.join(OUTPUT_DIR, f'predictions_seed_{seed}.csv')
    output_files.append(output_csv)
    # Run predict.py with the given seed and required drugs
    cmd = [
        'python', 'qm9_drug_material/predict.py',
        '--encoder_path', 'models/best_qm9_encoder.pth',
        '--material', 'CIF_files/Graphsene',
        '--sdf', '20220301-L1300-FDA-approved-Drug-Library.sdf',
        '--output', output_csv,
        '--required_drugs', required_drugs_arg
    ]
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    subprocess.run(cmd, env=env, check=True)

results = {drug: [] for drug in DRUGS}
for output_csv in output_files:
    if not os.path.exists(output_csv):
        print(f'Warning: expected output file {output_csv} not found; skipping')
        continue
    df = pd.read_csv(output_csv)

    # determine columns
    name_col = 'drug_name' if 'drug_name' in df.columns else ('name' if 'name' in df.columns else None)
    pred_col = 'prediction' if 'prediction' in df.columns else ('pred_energy' if 'pred_energy' in df.columns else None)

    for drug in DRUGS:
        # robust matching: normalize both sides
        if name_col is None or pred_col is None:
            continue
        matches = df[df[name_col].apply(normalize_name).str.contains(normalize_name(drug))]
        if not matches.empty:
            try:
                pred = float(matches.iloc[0][pred_col])
                results[drug].append(pred)
            except Exception:
                # skip non-numeric
                continue

# Compute average and median predictions and save summary
summary = []
for drug, preds in results.items():
    true_val = true_values.get(drug, None)
    if preds and true_val is not None:
        avg_pred = float(np.mean(preds))
        med_pred = float(np.median(preds))
        abs_error = float(abs(avg_pred - true_val))
        rmse = float(np.sqrt(np.mean([(p - true_val) ** 2 for p in preds])))
        summary.append({
            'drug': drug,
            'true_value': true_val,
            'avg_pred': avg_pred,
            'med_pred': med_pred,
            'mean_abs_error': abs_error,
            'rmse': rmse,
            'n_runs': len(preds)
        })
    else:
        summary.append({
            'drug': drug,
            'true_value': true_val,
            'avg_pred': None,
            'med_pred': None,
            'mean_abs_error': None,
            'rmse': None,
            'n_runs': len(preds)
        })

# Save summary to CSV inside OUTPUT_DIR
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'seed_prediction_summary.csv'), index=False)
print(summary_df)

# Print overall MAE and RMSE (averaged across drugs)
mae = summary_df['mean_abs_error'].dropna().mean() if 'mean_abs_error' in summary_df.columns else None
rmse = summary_df['rmse'].dropna().mean() if 'rmse' in summary_df.columns else None
if mae is not None:
    print(f'Overall Mean Absolute Error: {mae:.4f}')
if rmse is not None:
    print(f'Overall RMSE: {rmse:.4f}')
