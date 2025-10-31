import pandas as pd
import numpy as np
import os

# List of prediction CSVs (update if needed)
prediction_files = [f'predictions_seed_{seed}.csv' for seed in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]

# Drug names to analyze
DRUGS = [
    'Ifosfamide',
    'Pyrazinamide (Pyrazinoic acid amide)',
    'Zidovudine (Retrovir)',
    'Metformin hydrochloride (Glucophage)',
    'Captopril (Capoten)',
    'Tiopronin (Thiola)',
]

def normalize_name(name):
    return name.lower().replace('"', '').replace("'", '').replace(' ', '').strip()

# Use provided true values from attachment
true_values = {
    'Ifosfamide': 0.566,
    'Pyrazinamide (Pyrazinoic acid amide)': 0.39,
    'Zidovudine (Retrovir)': 0.36,
    'Metformin hydrochloride (Glucophage)': 0.94,
    'Captopril (Capoten)': 0.61,
    'Tiopronin (Thiola)': 0.38
}

# Collect predictions for each drug across runs
results = {drug: [] for drug in DRUGS}
for file in prediction_files:
    if not os.path.exists(file):
        print(f"Missing file: {file}")
        continue
    df = pd.read_csv(file)
    name_col = None
    if 'name' in df.columns:
        name_col = 'name'
    elif 'drug_name' in df.columns:
        name_col = 'drug_name'
    else:
        print(f"Warning: No drug name column found in {file}. Skipping.")
        continue
    df['norm_name'] = df[name_col].apply(normalize_name)
    for drug in DRUGS:
        norm_drug = normalize_name(drug)
        match = df[df['norm_name'] == norm_drug]
        if not match.empty:
            pred = match.iloc[0]['pred_energy']
            results[drug].append(pred)

# Compute average and median predictions
summary = []
for drug, preds in results.items():
    true_val = true_values[drug]
    if preds and true_val is not None:
        med_pred = np.median(preds)
        mean_pred = np.mean(preds)
        median_abs_error = abs(med_pred - true_val)
        mean_abs_error = abs(mean_pred - true_val)
        rmse = np.sqrt(np.mean([(med_pred - true_val)**2 for _ in preds]))
        summary.append({
            'drug': drug,
            'true_value': true_val,
            'med_pred': med_pred,
            'mean_pred': mean_pred,
            'median_abs_error': median_abs_error,
            'mean_abs_error': mean_abs_error,
            'rmse': rmse
        })
    else:
        summary.append({
            'drug': drug,
            'true_value': true_val,
            'med_pred': None,
            'mean_pred': None,
            'median_abs_error': None,
            'mean_abs_error': None,
            'rmse': None
        })

# Save summary to CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv('analyzed_seed_prediction_summary.csv', index=False)
print(summary_df)

# Print overall MAE and RMSE (using median and mean)
median_mae = summary_df['median_abs_error'].dropna().mean()
mean_mae = summary_df['mean_abs_error'].dropna().mean()
rmse = summary_df['rmse'].dropna().mean()
print(f'Overall Median Absolute Error: {median_mae:.4f}')
print(f'Overall Mean Absolute Error: {mean_mae:.4f}')
print(f'Overall RMSE (median): {rmse:.4f}')
