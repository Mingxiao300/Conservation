#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


"""
Create comparable three-panel coefficient plots (Fish/Wild/Domestic) from
two diagnostics CSVs:
  - outputs/diagnostics/beta_coefficients.csv      (Python/PyMC)
  - outputs/diagnostics/r_coefficients_by_outcome.csv (R/brms parsed)

Both plots share the same y-axis predictor ordering to enable direct visual
comparison. Each panel shows the posterior mean and 94% HDI (or 95% CI for R).

Outputs:
  - outputs/plots/effects_by_meat_type_from_python_csv.png
  - outputs/plots/effects_by_meat_type_from_r_csv.png
"""


def load_python_csv(path: str) -> pd.DataFrame:
    """Load Python beta coefficients CSV and normalize column names.
    Expected columns: ['mean','hdi_3%','hdi_97%','feature','outcome']
    """
    df = pd.read_csv(path)
    # Normalize column names if different casing exists
    cols = {c.lower(): c for c in df.columns}
    mean_col = cols.get('mean', 'mean')
    l_col = [c for c in df.columns if c.lower().startswith('hdi_3') or c.lower().startswith('l-95')]
    u_col = [c for c in df.columns if c.lower().startswith('hdi_97') or c.lower().startswith('u-95')]
    l_col = l_col[0] if l_col else 'hdi_3%'
    u_col = u_col[0] if u_col else 'hdi_97%'

    # Column with predictor name
    feature_col = 'feature' if 'feature' in df.columns else 'Predictor'
    outcome_col = 'outcome' if 'outcome' in df.columns else 'Outcome'

    # Standardize
    out = pd.DataFrame({
        'Outcome': df[outcome_col].map(lambda s: s.title() if isinstance(s, str) else s),
        'Predictor': df[feature_col],
        'mean': df[mean_col],
        'l': df[l_col],
        'u': df[u_col],
        'Source': 'Python',
    })
    return out


def load_r_csv(path: str) -> pd.DataFrame:
    """Load R coefficients CSV (from plot_effects_from_r_log.py) and normalize."""
    df = pd.read_csv(path)
    # Expected columns: Outcome, Predictor, mean, hdi_3%, hdi_97%
    l_col = [c for c in df.columns if c.lower().startswith(('hdi_3', 'l-95'))]
    u_col = [c for c in df.columns if c.lower().startswith(('hdi_97', 'u-95'))]
    l_col = l_col[0]
    u_col = u_col[0]
    out = pd.DataFrame({
        'Outcome': df['Outcome'],
        'Predictor': df['Predictor'],
        'mean': df['mean'],
        'l': df[l_col],
        'u': df[u_col],
        'Source': 'R',
    })
    return out


def prettify_labels(series: pd.Series) -> pd.Series:
    return (series
            .str.replace('_', ' ', regex=False)
            .str.replace('.', ' ', regex=False))


def compute_global_order(df_py: pd.DataFrame, df_r: pd.DataFrame) -> list:
    """Compute a single predictor order across both sources and outcomes.
    We use the average absolute mean over (source x outcome) and sort ascending
    (smallest effect at top, largest at bottom), to match prior plots.
    """
    combined = pd.concat([df_py[['Predictor', 'mean']], df_r[['Predictor', 'mean']]],
                         ignore_index=True)
    order_df = (combined.assign(abs_effect=lambda x: x['mean'].abs())
                        .groupby('Predictor')['abs_effect']
                        .mean()
                        .reset_index()
                        .sort_values(['abs_effect', 'Predictor'], ascending=[True, True]))
    order_labels = prettify_labels(order_df['Predictor'])
    return order_labels.tolist()


def plot_three_panel(df: pd.DataFrame, order_labels: list, title_prefix: str, out_path: str) -> None:
    sns.set(style='whitegrid', context='talk')
    outcomes = ['Fish', 'Wild', 'Domestic']
    fig, axes = plt.subplots(1, 3, figsize=(22, 12), sharey=True)
    palette = {
        'Increase': '#1f77b4',
        'Uncertain': '#8c8c8c',
        'Decrease': '#d62728',
    }
    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        sub = df[df['Outcome'] == outcome].copy()
        if sub.empty:
            ax.set_title(f'{outcome} Meat Effects')
            ax.axis('off')
            continue

        sub['PredictorLabel'] = prettify_labels(sub['Predictor'])
        sub['Direction'] = np.where((sub['l'] < 0) & (sub['u'] > 0), 'Uncertain',
                                    np.where(sub['mean'] > 0, 'Increase', 'Decrease'))

        sns.barplot(
            data=sub,
            x='mean', y='PredictorLabel', hue='Direction',
            dodge=False, palette=palette, ax=ax, order=order_labels
        )
        # HDI whiskers
        for yi, (_, row) in enumerate(sub.set_index('PredictorLabel').loc[order_labels].dropna(how='any').iterrows()):
            ax.plot([row['l'], row['u']], [yi, yi], color='black', linewidth=2, alpha=0.8)

        ax.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_xlabel('Change in expected consumption (log scale)')
        ax.set_ylabel('' if i > 0 else 'Predictor')
        ax.set_title(f'{title_prefix}: {outcome} Meat Effects')
        ax.legend().remove()

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    order = ['Increase', 'Uncertain', 'Decrease']
    handles_ordered = [h for l, h in [(l, handles[labels.index(l)]) for l in order if l in labels]]
    labels_ordered = [l for l in order if l in labels]
    if handles_ordered:
        fig.legend(handles_ordered, labels_ordered, title='Effect direction',
                   loc='center right', bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create comparable coefficient plots from Python and R CSVs')
    parser.add_argument('--python_csv', default='outputs/diagnostics/beta_coefficients.csv',
                        help='Path to Python beta coefficients CSV')
    parser.add_argument('--r_csv', default='outputs/diagnostics/r_coefficients_by_outcome.csv',
                        help='Path to R coefficients CSV parsed from logs')
    parser.add_argument('--out_dir', default='outputs/plots', help='Directory to save plots')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    py_csv = os.path.join(base_dir, args.python_csv)
    r_csv = os.path.join(base_dir, args.r_csv)
    out_dir = os.path.join(base_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(py_csv):
        print(f"❌ Python CSV not found: {py_csv}")
        sys.exit(1)
    if not os.path.exists(r_csv):
        print(f"❌ R CSV not found: {r_csv}")
        sys.exit(1)

    df_py = load_python_csv(py_csv)
    df_r = load_r_csv(r_csv)

    # Ensure consistent outcome labels
    normalize_outcome = {'fish': 'Fish', 'wild': 'Wild', 'domestic': 'Domestic'}
    df_py['Outcome'] = df_py['Outcome'].replace(normalize_outcome)
    df_r['Outcome'] = df_r['Outcome'].replace(normalize_outcome)

    # Restrict to known outcomes
    keep = ['Fish', 'Wild', 'Domestic']
    df_py = df_py[df_py['Outcome'].isin(keep)]
    df_r = df_r[df_r['Outcome'].isin(keep)]

    # Compute a single predictor order used by BOTH plots
    order_labels = compute_global_order(df_py, df_r)

    # Plot for Python coefficients
    out_py = os.path.join(out_dir, 'effects_by_meat_type_from_python_csv.png')
    plot_three_panel(df_py, order_labels, 'Python', out_py)
    print(f'✅ Python plot saved to {out_py}')

    # Plot for R coefficients
    out_r = os.path.join(out_dir, 'effects_by_meat_type_from_r_csv.png')
    plot_three_panel(df_r, order_labels, 'R', out_r)
    print(f'✅ R plot saved to {out_r}')

    # Also emit the unified order for audit
    order_out = os.path.join(out_dir, 'predictor_order_used.txt')
    with open(order_out, 'w') as f:
        for label in order_labels:
            f.write(f"{label}\n")
    print(f'ℹ️  Predictor order written to {order_out}')


if __name__ == '__main__':
    sys.exit(main())


