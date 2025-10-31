"""
seeds_detailed_analysis.py

Reads prediction CSVs named like `predictions_seed_<seed>.csv` and computes:
- Per-seed metrics: mean absolute error (MAE), median absolute error, RMSE across target drugs
- Per-drug summary across seeds: median, mean, std of predictions, MAE vs provided true values
- Saves `per_seed_metrics.csv` and `per_drug_summary.csv`
- Optionally saves simple plots if matplotlib is available

Run: python seeds_detailed_analysis.py
"""

try:

    import glob
    import os
    import re
    import pandas as pd
    import numpy as np

    # --- Configuration ---
    PRED_PATTERN = 'predictions_seed_*.csv'
    PLOT_DIR = 'seed_analysis_plots'
    # Save CSV outputs into the same results folder
    OUT_PER_SEED = os.path.join(PLOT_DIR, 'per_seed_metrics.csv')
    OUT_PER_DRUG = os.path.join(PLOT_DIR, 'per_drug_summary.csv')

    # Target drugs and provided true values (from your attachment)
    DRUGS = [
        'Ifosfamide',
        'Pyrazinamide (Pyrazinoic acid amide)',
        'Zidovudine (Retrovir)',
        'Metformin hydrochloride (Glucophage)',
        'Captopril (Capoten)',
        'Tiopronin (Thiola)',
    ]
    TRUE_VALUES = {
        'Ifosfamide': 0.566,
        'Pyrazinamide (Pyrazinoic acid amide)': 0.397,
        'Zidovudine (Retrovir)': 0.3645,
        'Metformin hydrochloride (Glucophage)': 0.943,
        'Captopril (Capoten)': 0.615,
        'Tiopronin (Thiola)': 0.388
    }

    def normalize_name(name):
        if pd.isna(name):
            return ''
        return str(name).lower().replace('"', '').replace("'", '').replace(' ', '').strip()

    # CLI: allow analyzing a directory of prediction CSVs or a specific list of seeds
    import argparse
    parser = argparse.ArgumentParser(description='Analyze per-seed prediction CSVs')
    parser.add_argument('--pred_dir', type=str, default='.', help='Directory containing prediction CSVs')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated list of seed IDs to analyze (overrides scanning directory)')
    args = parser.parse_args()

    # Determine files to analyze
    if args.seeds:
        seed_list = [s.strip() for s in args.seeds.split(',') if s.strip()]
        files = [os.path.join(args.pred_dir, f'predictions_seed_{s}.csv') for s in seed_list]
    else:
        files = sorted(glob.glob(os.path.join(args.pred_dir, PRED_PATTERN)))
    if not files:
        print(f'No prediction files found (dir={args.pred_dir}, pattern={PRED_PATTERN})')


    # (files list already determined from CLI args above)

    # Prepare containers
    per_seed_rows = []
    per_drug_preds = {drug: [] for drug in DRUGS}

    # Process files
    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.search(r'predictions_seed_(\d+)\.csv', fname)
        if m:
            seed = int(m.group(1))
        else:
            seed = fname

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f'Could not read {fpath}: {e}')
            continue

        # find name column
        if 'name' in df.columns:
            name_col = 'name'
        elif 'drug_name' in df.columns:
            name_col = 'drug_name'
        else:
            print(f'Warning: No drug name column found in {fpath}. Skipping.')
            continue

        # normalize
        df['norm_name'] = df[name_col].apply(normalize_name)

        # collect predictions for target drugs
        seed_errors = []
        seed_abs_errors = []
        seed_available = 0
        for drug in DRUGS:
            norm_drug = normalize_name(drug)
            match = df[df['norm_name'] == norm_drug]
            if not match.empty:
                pred = float(match.iloc[0]['pred_energy'])
                per_drug_preds[drug].append((seed, pred))
                true = TRUE_VALUES.get(drug, None)
                if true is not None:
                    err = pred - true
                    seed_errors.append(err)
                    seed_abs_errors.append(abs(err))
                seed_available += 1
            else:
                # missing prediction for this drug in this seed
                pass

        # compute per-seed metrics if we have any errors
        if seed_available > 0 and seed_errors:
            mae = float(np.mean(seed_abs_errors))
            median_ae = float(np.median(seed_abs_errors))
            rmse = float(np.sqrt(np.mean(np.square(seed_errors))))
        else:
            mae = np.nan
            median_ae = np.nan
            rmse = np.nan

        per_seed_rows.append({
            'seed': seed,
            'n_predicted_drugs': seed_available,
            'mean_abs_error': mae,
            'median_abs_error': median_ae,
            'rmse': rmse
        })


    # Save per-seed metrics
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(by='seed')
    # ensure output directory exists before saving CSVs/plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    per_seed_df.to_csv(OUT_PER_SEED, index=False)
    print(f'Saved per-seed metrics to {OUT_PER_SEED}')


    # Compute per-drug summary across seeds
    rows = []
    for drug, seed_preds in per_drug_preds.items():
        preds = [p for (s, p) in seed_preds]
        true = TRUE_VALUES.get(drug, None)
        if preds:
            median_pred = float(np.median(preds))
            mean_pred = float(np.mean(preds))
            std_pred = float(np.std(preds, ddof=0))
            n = len(preds)
            mae = float(abs(mean_pred - true)) if true is not None else np.nan
            median_ae = float(abs(median_pred - true)) if true is not None else np.nan
        else:
            median_pred = mean_pred = std_pred = n = np.nan
            mae = median_ae = np.nan

        rows.append({
            'drug': drug,
            'true_value': true,
            'n_seeds': n,
            'median_pred': median_pred,
            'mean_pred': mean_pred,
            'std_pred': std_pred,
            'median_abs_error': median_ae,
            'mean_abs_error': mae
        })

    per_drug_df = pd.DataFrame(rows)
    per_drug_df.to_csv(OUT_PER_DRUG, index=False)
    print(f'Saved per-drug summary to {OUT_PER_DRUG}')

    # Print top/bottom seeds by MAE
    if not per_seed_df['mean_abs_error'].dropna().empty:
        best = per_seed_df.sort_values('mean_abs_error').head(5)
        worst = per_seed_df.sort_values('mean_abs_error', ascending=False).head(5)
        print('\nBest seeds by mean abs error:')
        print(best)
        print('\nWorst seeds by mean abs error:')
        print(worst)


    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        # set Times New Roman and increase font size for all plots
        try:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'font.size': 14,
            })
        except Exception:
            pass
        os.makedirs(PLOT_DIR, exist_ok=True)

        # Per-seed MAE histogram (plot both mean and median absolute error)
        plt.figure()
        mean_vals = per_seed_df['mean_abs_error'].dropna() if 'mean_abs_error' in per_seed_df.columns else pd.Series(dtype=float)
        median_vals = per_seed_df['median_abs_error'].dropna() if 'median_abs_error' in per_seed_df.columns else pd.Series(dtype=float)
        if mean_vals.empty and median_vals.empty:
            plt.text(0.5, 0.5, 'No MAE data available', ha='center', va='center')
        else:
            bins = 20
            data = []
            labels = []
            colors = []
            if not mean_vals.empty:
                data.append(mean_vals)
                labels.append('mean MAE')
                colors.append('tab:blue')
            if not median_vals.empty:
                data.append(median_vals)
                labels.append('median AE')
                colors.append('tab:orange')
            # use stacked=False to overlay histograms with transparency
            plt.hist(data, bins=bins, label=labels, color=colors, alpha=0.6)
            plt.legend()

            plt.xlim(0,0.6)

        # plt.title('Per-seed Absolute Error Distribution')
        plt.xlabel('Absolute error')
        plt.ylabel('Count')
        plt.savefig(os.path.join(PLOT_DIR, 'per_seed_mae_hist_mean_median.png'))

        # Per-seed MAE scatter (by seed)
        plt.figure(figsize=(10,4))
        df_plot = per_seed_df.dropna(subset=['mean_abs_error']).sort_values('seed')
        plt.scatter(df_plot['seed'], df_plot['mean_abs_error'], color='tab:blue')
        med_line = df_plot['mean_abs_error'].median()
        mean_line = df_plot['mean_abs_error'].mean()
        plt.axhline(med_line, color='tab:orange', linestyle='--', label=f'median MAE={med_line:.3f}')
        plt.axhline(mean_line, color='tab:green', linestyle=':', label=f'mean MAE={mean_line:.3f}')
        plt.title('Per-seed Mean Absolute Error')
        plt.xlabel('Seed')
        plt.ylabel('Mean absolute error')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'per_seed_mae_scatter.png'))

        # Per-drug boxplot of predictions with true value lines, mean & median markers and counts
        plt.figure(figsize=(12,6))
        data = [ [p for (s,p) in per_drug_preds[d]] for d in DRUGS ]
        # create boxplot (we'll set xtick labels separately after cleaning them)
        bp = plt.boxplot(data, showmeans=False, patch_artist=True)
        # style boxes
        for box in bp['boxes']:
            box.set(facecolor='lightblue',  edgecolor='black')

        # overlay mean and median markers and true value
        # compute ylimits to place count annotations
        all_vals = [v for sub in data for v in sub] if any(len(sub)>0 for sub in data) else [0]
        ymin = min(all_vals) - 0.1 * (max(all_vals) - min(all_vals) if max(all_vals)>min(all_vals) else 1)
        ymax = max(all_vals) + 0.1 * (max(all_vals) - min(all_vals) if max(all_vals)>min(all_vals) else 1)
        plt.ylim(ymin, ymax)
        for i, drug in enumerate(DRUGS, start=1):
            preds = [p for (s,p) in per_drug_preds[drug]]
            if not preds:
                continue
            median_pred = np.median(preds)
            mean_pred = np.mean(preds)
            # median marker
            plt.plot(i, median_pred, marker='D', color='tab:orange', markersize=6, label='median' if i==1 else '')
            # mean marker
            plt.plot(i, mean_pred, marker='o', color='tab:green', markersize=5, label='mean' if i==1 else '')
            # true value line
            true = TRUE_VALUES.get(drug, None)
            if true is not None:
                plt.plot([i-0.35, i+0.35], [true, true], color='red', linewidth=1.5, alpha=0.8, label='true value' if i==1 else '')
        # (removed per-drug n annotation for cleaner plot)

        plt.title('Per-drug predictions across seeds')
        plt.ylabel('Predicted value')
        # set full drug names as tick labels centered under boxes
        # remove any bracketed text like "(...)" or "[...]" from labels for a cleaner plot
        def _clean_label(s):
            lab = re.sub(r"[\(\[].*?[\)\]]", '', s)
            lab = ' '.join(lab.split()).strip()
            if len(lab) > 40:
                return lab[:37] + '...'
            return lab

        labels = [_clean_label(d) for d in DRUGS]
        # Use letter labels A,B,C... on the x-axis to anonymize drug names in the plot
        letters = [chr(ord('A') + i) for i in range(len(labels))]
        # save mapping for reference
        mapping = [{'letter': letters[i], 'drug': DRUGS[i], 'clean_label': labels[i]} for i in range(len(labels))]
        try:
            pd.DataFrame(mapping).to_csv(os.path.join(PLOT_DIR, 'drug_label_mapping.csv'), index=False)
        except Exception:
            pass
        print('Drug label mapping:')
        for m in mapping:
            print(f"{m['letter']} -> {m['drug']}")
        ax = plt.gca()
        ax.set_xticks(range(1, len(letters) + 1))
        ax.set_xticklabels(letters, rotation=0, ha='center')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'per_drug_boxplot_enhanced.png'))

        # Outlier detection per drug (1.5*IQR rule); save list
        outlier_rows = []
        for drug in DRUGS:
            preds = [(s,p) for (s,p) in per_drug_preds[drug]]
            if not preds:
                continue
            values = np.array([p for (_,p) in preds])
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            for (s,p) in preds:
                if p < lower or p > upper:
                    outlier_rows.append({'drug': drug, 'seed': s, 'prediction': p, 'lower': lower, 'upper': upper})

        if outlier_rows:
            outlier_df = pd.DataFrame(outlier_rows)
            outlier_df.to_csv(os.path.join(PLOT_DIR, 'outlier_list.csv'), index=False)
            print(f'Saved outlier list to {os.path.join(PLOT_DIR, "outlier_list.csv")}')

        print(f'Saved plots to {PLOT_DIR}/')
    except Exception as e:
        print('Matplotlib not available or plotting failed:', e)

    print('\nDone.')
           
    # Save per-seed metrics
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(by='seed')
    per_seed_df.to_csv(OUT_PER_SEED, index=False)
    print(f'Saved per-seed metrics to {OUT_PER_SEED}')

    # Compute per-drug summary across seeds
    rows = []
    for drug, seed_preds in per_drug_preds.items():
        preds = [p for (s, p) in seed_preds]
        true = TRUE_VALUES.get(drug, None)
        if preds:
            median_pred = float(np.median(preds))
            mean_pred = float(np.mean(preds))
            std_pred = float(np.std(preds, ddof=0))
            n = len(preds)
            mae = float(abs(mean_pred - true)) if true is not None else np.nan
            median_ae = float(abs(median_pred - true)) if true is not None else np.nan
        else:
            median_pred = mean_pred = std_pred = n = np.nan
            mae = median_ae = np.nan

        rows.append({
            'drug': drug,
            'true_value': true,
            'n_seeds': n,
            'median_pred': median_pred,
            'mean_pred': mean_pred,
            'std_pred': std_pred,
            'median_abs_error': median_ae,
            'mean_abs_error': mae
        })

    per_drug_df = pd.DataFrame(rows)
    per_drug_df.to_csv(OUT_PER_DRUG, index=False)
    print(f'Saved per-drug summary to {OUT_PER_DRUG}')

    # Print top/bottom seeds by MAE
    if not per_seed_df['mean_abs_error'].dropna().empty:
        best = per_seed_df.sort_values('mean_abs_error').head(5)
        worst = per_seed_df.sort_values('mean_abs_error', ascending=False).head(5)
        print('\nBest seeds by mean abs error:')
        print(best)
        print('\nWorst seeds by mean abs error:')
        print(worst)

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        # set Times New Roman and increase font size for all plots
        try:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'font.size': 20,
            })
        except Exception:
            pass
        os.makedirs(PLOT_DIR, exist_ok=True)

        # Per-seed MAE histogram
        plt.figure()
        per_seed_df['mean_abs_error'].dropna().hist(bins=20)
        plt.title('Per-seed MAE')
        plt.xlabel('Mean absolute error')
        plt.ylabel('Count')
        plt.savefig(os.path.join(PLOT_DIR, 'per_seed_mae_hist.png'))

        # Per-seed MAE scatter (by seed)
        plt.figure(figsize=(10,4))
        df_plot = per_seed_df.dropna(subset=['mean_abs_error']).sort_values('seed')
        plt.scatter(df_plot['seed'], df_plot['mean_abs_error'], color='tab:blue')
        med_line = df_plot['mean_abs_error'].median()
        mean_line = df_plot['mean_abs_error'].mean()
        plt.axhline(med_line, color='tab:orange', linestyle='--', label=f'median MAE={med_line:.3f}')
        plt.axhline(mean_line, color='tab:green', linestyle=':', label=f'mean MAE={mean_line:.3f}')
        # plt.title('Per-seed Mean Absolute Error')
        plt.xlabel('Seed')
        plt.ylabel('Mean absolute error')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'per_seed_mae_scatter.png'))

        # Per-drug boxplot of predictions with true value lines, mean & median markers and counts
        plt.figure(figsize=(12,6))
        data = [ [p for (s,p) in per_drug_preds[d]] for d in DRUGS ]
        # create boxplot (labels cleaned and set after)
        bp = plt.boxplot(data, showmeans=False, widths=0.4, patch_artist=True)
        # style boxes
        for box in bp['boxes']:
            box.set(facecolor='lightblue', edgecolor='black')

        # overlay mean and median markers and true value
        # compute ylimits to place count annotations
        all_vals = [v for sub in data for v in sub] if any(len(sub)>0 for sub in data) else [0]
        ymin = min(all_vals) - 0.1 * (max(all_vals) - min(all_vals) if max(all_vals)>min(all_vals) else 1)
        ymax = max(all_vals) + 0.1 * (max(all_vals) - min(all_vals) if max(all_vals)>min(all_vals) else 1)
        plt.ylim(ymin, ymax)
        for i, drug in enumerate(DRUGS, start=1):
            preds = [p for (s,p) in per_drug_preds[drug]]
            if not preds:
                continue
            median_pred = np.median(preds)
            mean_pred = np.mean(preds)
            # median marker
            plt.plot(i, median_pred, marker='D', color='tab:orange', markersize=6, label='median' if i==1 else '')
            # mean marker
            plt.plot(i, mean_pred, marker='o', color='tab:green', markersize=5, label='mean' if i==1 else '')
            # true value line
            true = TRUE_VALUES.get(drug, None)
            if true is not None:
                plt.plot([i-0.35, i+0.35], [true, true], color='red', linewidth=1.5, alpha=0.8, label='true value' if i==1 else '')
        # (removed per-drug n annotation for cleaner plot)

        plt.title('Per-drug predictions across seeds')
        plt.ylabel('Predicted value')
        # clean bracketed content and set tick labels
        def _clean_label(s):
            lab = re.sub(r"[\(\[].*?[\)\]]", '', s)
            lab = ' '.join(lab.split()).strip()
            if len(lab) > 40:
                return lab[:37] + '...'
            return lab

        labels = [_clean_label(d) for d in DRUGS]
        # Use letter labels A,B,C... on the x-axis to anonymize drug names in the plot
        letters = [chr(ord('A') + i) for i in range(len(labels))]
        # save mapping for reference (overwrite if exists)
        mapping = [{'letter': letters[i], 'drug': DRUGS[i], 'clean_label': labels[i]} for i in range(len(labels))]
        try:
            pd.DataFrame(mapping).to_csv(os.path.join(PLOT_DIR, 'drug_label_mapping.csv'), index=False)
        except Exception:
            pass
        print('Drug label mapping:')
        for m in mapping:
            print(f"{m['letter']} -> {m['drug']}")
        ax = plt.gca()
        ax.set_xticks(range(1, len(letters) + 1))
        ax.set_xticklabels(letters, rotation=0, ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'per_drug_boxplot_enhanced.png'))

        # Outlier detection per drug (1.5*IQR rule); save list
        outlier_rows = []
        for drug in DRUGS:
            preds = [(s,p) for (s,p) in per_drug_preds[drug]]
            if not preds:
                continue
            values = np.array([p for (_,p) in preds])
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            for (s,p) in preds:
                if p < lower or p > upper:
                    outlier_rows.append({'drug': drug, 'seed': s, 'prediction': p, 'lower': lower, 'upper': upper})

        if outlier_rows:
            outlier_df = pd.DataFrame(outlier_rows)
            outlier_df.to_csv(os.path.join(PLOT_DIR, 'outlier_list.csv'), index=False)
            print(f'Saved outlier list to {os.path.join(PLOT_DIR, "outlier_list.csv")}')

        print(f'Saved plots to {PLOT_DIR}/')
    except Exception as e:
        print('Matplotlib not available or plotting failed:', e)

    print('\nDone.')

except Exception as e:
    print('Matplotlib not available or plotting failed:', e)
