#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

"""
Generate a three-panel coefficient plot (Fish, Wild, Domestic) from
outputs/diagnostics/beta_coefficients.csv produced by fit_model_pymc.py.

Usage:
  python3 plot_effects_by_meat_type.py [BASE_DIR]

Defaults:
  BASE_DIR = directory of this script
Reads:
  {BASE_DIR}/outputs/diagnostics/beta_coefficients.csv
Writes:
  {BASE_DIR}/outputs/plots/effects_by_meat_type.png
"""

def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
    diagnostics_dir = os.path.join(base_dir, 'outputs', 'diagnostics')
    plots_dir = os.path.join(base_dir, 'outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    beta_csv = os.path.join(diagnostics_dir, 'beta_coefficients.csv')
    if not os.path.exists(beta_csv):
        print(f"❌ beta_coefficients.csv not found at {beta_csv}")
        print("  Run the multivariate Python model first to generate diagnostics.")
        sys.exit(1)

    df = pd.read_csv(beta_csv)
    # Expected columns include: mean, hdi_3%, hdi_97%, and we created 'feature' and 'outcome'
    # Harmonize column names just in case
    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    required_cols = {'mean', 'hdi_3%', 'hdi_97%', 'feature', 'outcome'}
    if not required_cols.issubset(df.columns):
        print(f"❌ Missing required columns in {beta_csv}. Found: {list(df.columns)}")
        sys.exit(1)

    # Only keep fixed effects (ignore random effects if present)
    # Heuristic: outcome is one of Fish, Wild, Domestic
    df = df[df['outcome'].isin(['Fish', 'Wild', 'Domestic'])].copy()

    # Compute direction and absolute effect for sorting
    def direction_from_row(row):
        if row['hdi_3%'] < 0 and row['hdi_97%'] > 0:
            return 'Uncertain'
        return 'Increase' if row['mean'] > 0 else 'Decrease'

    df['Direction'] = df.apply(direction_from_row, axis=1)
    df['abs_effect'] = df['mean'].abs()

    # Sort by absolute effect within each outcome for nicer display
    df_sorted = (
        df.sort_values(['outcome', 'abs_effect'], ascending=[True, True])
    )

    # Plot
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=False)

    palette = {
        'Increase': '#1f77b4',
        'Uncertain': '#8c8c8c',
        'Decrease': '#d62728'
    }

    outcomes = ['Fish', 'Wild', 'Domestic']
    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        sub = df_sorted[df_sorted['outcome'] == outcome].copy()
        if sub.empty:
            ax.set_title(f"{outcome} Meat Effects")
            ax.axis('off')
            continue

        # Order predictors by absolute effect
        sub = sub.sort_values('abs_effect', ascending=True)
        # Replace underscores for readability
        sub['Predictor'] = sub['feature'].str.replace('_', ' ', regex=False)

        sns.barplot(
            data=sub,
            x='mean', y='Predictor', hue='Direction', dodge=False,
            palette=palette, ax=ax
        )
        ax.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_xlabel('Change in expected consumption (log scale)')
        ax.set_ylabel('')
        ax.set_title(f"{outcome} Meat Effects")
        ax.legend().remove()

    # Build a single legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    order = ['Increase', 'Uncertain', 'Decrease']
    handles_ordered = [handles[labels.index(l)] for l in order if l in labels]
    labels_ordered = [l for l in order if l in labels]
    if handles_ordered:
        fig.legend(handles_ordered, labels_ordered, title='Effect direction',
                   loc='center right', bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    out_path = os.path.join(plots_dir, 'effects_by_meat_type.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Effects plot saved to {out_path}")

if __name__ == '__main__':
    sys.exit(main())
