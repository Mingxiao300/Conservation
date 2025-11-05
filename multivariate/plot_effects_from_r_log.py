#!/usr/bin/env python3
import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

"""
Parse an R brms log file (Regression Coefficients table) and create a three-panel
coefficient effects plot for Fish, Wild, Domestic, similar to the R plot shown.

Usage:
  python3 plot_effects_from_r_log.py /path/to/r_YYYYMMDD_HHMMSS.log
  # or run without arguments to auto-pick the latest r_*.log in logs/

Outputs:
  - outputs/diagnostics/r_coefficients_by_outcome.csv
  - outputs/plots/effects_by_meat_type_from_r.png
"""

ROW_RE = re.compile(
    r"^(?P<name>\S+)\s+"  # parameter name
    r"(?P<estimate>-?\d+\.\d+)\s+"  # Estimate
    r"(?P<stderr>-?\d+\.\d+)\s+"  # Est.Error
    r"(?P<l95>-?\d+\.\d+)\s+"      # l-95% CI
    r"(?P<u95>-?\d+\.\d+)\s+"      # u-95% CI
    r"(?P<rhat>\d+\.\d+)\s*$"      # Rhat
)

PREFIX_TO_OUTCOME = {
    'logfish_': 'Fish',
    'logwild_': 'Wild',
    'logdomestic_': 'Domestic',
}


def find_latest_r_log(base_dir: str) -> str:
    logs_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(logs_dir):
        return ''
    candidates = [
        os.path.join(logs_dir, f) for f in os.listdir(logs_dir)
        if f.startswith('r_') and f.endswith('.log')
    ]
    if not candidates:
        return ''
    return max(candidates, key=os.path.getmtime)


def parse_coefficients_from_log(log_path: str) -> pd.DataFrame:
    rows = []
    in_table = False
    with open(log_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not in_table:
                if line.strip().startswith('Regression Coefficients:'):
                    in_table = True
                continue
            # Stop parsing at Bulk_ESS header or empty line after table
            if 'Bulk_ESS' in line:
                break
            if not line.strip():
                continue
            if line.strip().startswith(('Estimate',)):
                # header row
                continue
            m = ROW_RE.match(line)
            if m:
                name = m.group('name')
                est = float(m.group('estimate'))
                l95 = float(m.group('l95'))
                u95 = float(m.group('u95'))
                rhat = float(m.group('rhat'))
                # Map prefix to outcome
                outcome = None
                predictor = name
                for prefix, out in PREFIX_TO_OUTCOME.items():
                    if name.startswith(prefix):
                        outcome = out
                        predictor = name[len(prefix):]
                        break
                if outcome is None:
                    continue
                if predictor.lower() == 'intercept':
                    # Skip intercepts in the effects plot
                    continue
                rows.append({
                    'Outcome': outcome,
                    'Predictor': predictor,
                    'mean': est,
                    'hdi_3%': l95,
                    'hdi_97%': u95,
                    'Rhat': rhat,
                })
    if not rows:
        raise ValueError('No coefficient rows parsed from log. Check the file and section.')
    df = pd.DataFrame(rows)
    return df


def build_direction(df: pd.DataFrame) -> pd.DataFrame:
    def direction(row):
        if row['hdi_3%'] < 0 and row['hdi_97%'] > 0:
            return 'Uncertain'
        return 'Increase' if row['mean'] > 0 else 'Decrease'
    df['Direction'] = df.apply(direction, axis=1)
    df['abs_effect'] = df['mean'].abs()
    return df


def compute_global_order(df: pd.DataFrame) -> list:
    """Compute a global predictor order (same across panels) by overall effect size.
    We use the average absolute effect across outcomes; ties broken alphabetically.
    """
    order_df = (
        df.groupby('Predictor')['abs_effect']
          .mean()
          .reset_index()
          .sort_values(['abs_effect', 'Predictor'], ascending=[True, True])
    )
    # Make pretty labels (replace '_' and '.')
    order_df['Label'] = (
        order_df['Predictor']
        .str.replace('_', ' ', regex=False)
        .str.replace('.', ' ', regex=False)
    )
    return order_df['Label'].tolist()


def plot_effects(df: pd.DataFrame, out_path: str) -> None:
    sns.set(style='whitegrid', context='talk')

    fig, axes = plt.subplots(1, 3, figsize=(22, 12), sharey=True)
    palette = {
        'Increase': '#1f77b4',
        'Uncertain': '#8c8c8c',
        'Decrease': '#d62728',
    }
    outcomes = ['Fish', 'Wild', 'Domestic']
    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        sub = df[df['Outcome'] == outcome].copy()
        if sub.empty:
            ax.set_title(f'{outcome} Meat Effects')
            ax.axis('off')
            continue
        sub['PredictorLabel'] = (
            sub['Predictor']
            .str.replace('_', ' ', regex=False)
            .str.replace('.', ' ', regex=False)
        )
        # Order predictors within each panel by absolute effect size (ascending),
        # to match the ordering style of the attached reference plot
        sub_sorted = sub.sort_values('abs_effect', ascending=True)
        panel_order = sub_sorted['PredictorLabel'].tolist()
        # Ensure consistent categorical ordering across panels
        sns.barplot(
            data=sub,
            x='mean', y='PredictorLabel', hue='Direction',
            dodge=False, palette=palette, ax=ax, order=panel_order
        )
        ax.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax.set_xlabel('Change in expected consumption')
        ax.set_ylabel('' if i > 0 else 'Predictor')
        ax.set_title(f'{outcome} Meat Effects')
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


def plot_sign_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Create a coefficient sign/value heatmap (Predictor x Outcome).
    Values are mean coefficients; blue=negative, red=positive, centered at 0.
    """
    # Build wide matrix (Predictor x Outcome)
    mat = (
        df[['Predictor', 'Outcome', 'mean']]
          .pivot(index='Predictor', columns='Outcome', values='mean')
          .fillna(0.0)
    )
    # Nice labels
    mat.index = (
        pd.Series(mat.index)
          .str.replace('_', ' ', regex=False)
          .str.replace('.', ' ', regex=False)
          .values
    )
    # Order rows by overall absolute effect (descending for readability)
    order = np.argsort(-np.abs(mat).mean(axis=1).values)
    mat = mat.iloc[order]

    plt.figure(figsize=(10, 12))
    sns.set(style='white', context='talk')
    ax = sns.heatmap(
        mat,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0.0,
        cbar_kws={'label': 'Coefficient Value'}
    )
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Predictor')
    ax.set_title('Coefficient Sign by Predictor and Outcome (R coefficients)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = os.path.dirname(__file__)
    diagnostics_dir = os.path.join(base_dir, 'outputs', 'diagnostics')
    plots_dir = os.path.join(base_dir, 'outputs', 'plots')
    os.makedirs(diagnostics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = find_latest_r_log(base_dir)
    if not log_path or not os.path.exists(log_path):
        print('❌ R log file not found. Provide the path as an argument, e.g.:')
        print('   python3 plot_effects_from_r_log.py logs/r_YYYYMMDD_HHMMSS.log')
        sys.exit(1)

    print(f'Parsing coefficients from: {log_path}')
    df = parse_coefficients_from_log(log_path)
    df = build_direction(df)

    # Save coefficients CSV
    csv_out = os.path.join(diagnostics_dir, 'r_coefficients_by_outcome.csv')
    df.to_csv(csv_out, index=False)
    print(f'✅ Coefficients saved to {csv_out}')

    # Plot: effects panels
    plot_out = os.path.join(plots_dir, 'effects_by_meat_type_from_r.png')
    plot_effects(df, plot_out)
    print(f'✅ Plot saved to {plot_out}')

    # Plot: coefficient sign/value heatmap
    heatmap_out = os.path.join(plots_dir, 'coefficient_sign_heatmap_from_r.png')
    plot_sign_heatmap(df, heatmap_out)
    print(f'✅ Heatmap saved to {heatmap_out}')

if __name__ == '__main__':
    sys.exit(main())
