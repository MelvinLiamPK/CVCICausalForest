#!/usr/bin/env python3
"""
Post-run helper for Sherlock CVCI-CF results.

Checks which experiments completed, prints combined summary tables,
and creates a combined multi-panel figure.

Usage:
    python sherlock_collect.py /path/to/results     # Check and summarize
    python sherlock_collect.py /path/to/results --plot  # Also create combined plot
"""

import os
import sys
import json
import argparse
import glob


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def check_results(results_dir):
    """Scan results directory and report what completed."""
    expected = [
        ('epsilon', 'constant'), ('epsilon', 'step'), ('epsilon', 'nonlinear'),
        ('nobs', 'constant'), ('nobs', 'step'), ('nobs', 'nonlinear'),
        ('confounding', 'constant'), ('confounding', 'step'), ('confounding', 'nonlinear'),
    ]

    print(f"\nScanning: {results_dir}")
    print(f"{'='*60}")

    found = {}
    missing = []

    for axis, cate in expected:
        pattern = os.path.join(results_dir, f'{axis}_{cate}_*')
        matches = sorted(glob.glob(pattern))

        if matches:
            # Use the most recent one
            exp_dir = matches[-1]
            meta_path = os.path.join(exp_dir, 'metadata.json')
            summary_path = os.path.join(exp_dir, 'summary.json')
            results_path = os.path.join(exp_dir, 'results_all.json')

            status = []
            if os.path.exists(meta_path):
                status.append('meta')
            if os.path.exists(results_path):
                status.append('results')
            if os.path.exists(summary_path):
                status.append('summary')

            n_sims = '?'
            if os.path.exists(summary_path):
                summary = load_json(summary_path)
                if summary:
                    n_sims = summary[0].get('n_sims', '?')

            found[(axis, cate)] = exp_dir
            status_str = ', '.join(status)
            print(f"  OK   {axis:>12} x {cate:<12} | {n_sims} sims | [{status_str}]")
        else:
            missing.append((axis, cate))
            print(f"  MISS {axis:>12} x {cate:<12}")

    print(f"\n  Found: {len(found)}/9  |  Missing: {len(missing)}/9")

    if missing:
        print(f"\n  Missing experiments:")
        for axis, cate in missing:
            print(f"    python cf_simulations.py --axis {axis} --cate {cate}")

    return found


def print_combined_summary(found):
    """Print a combined table across all experiments."""
    print(f"\n{'='*80}")
    print("COMBINED SUMMARY")
    print(f"{'='*80}")

    for (axis, cate), exp_dir in sorted(found.items()):
        summary_path = os.path.join(exp_dir, 'summary.json')
        if not os.path.exists(summary_path):
            continue

        summary = load_json(summary_path)
        if not summary:
            continue

        # Determine parameter name
        if axis == 'epsilon':
            param_name = 'epsilon'
        elif axis == 'nobs':
            param_name = 'n_obs'
        else:
            param_name = 'confounding'

        print(f"\n--- {axis} x {cate} ---")
        print(f"{'Param':>10} | {'Exp CATE':>10} | {'Obs CATE':>10} | "
              f"{'Pool CATE':>10} | {'CVCI CATE':>10} | {'lam*':>6} | "
              f"{'CVCI best?':>10}")
        print("-" * 88)

        for row in summary:
            pv = row.get(param_name, 0)
            exp_c = row.get('exp_only_cate_mse_mean', float('nan'))
            obs_c = row.get('obs_only_cate_mse_mean', float('nan'))
            pool_c = row.get('pooled_cate_mse_mean', float('nan'))
            cvci_c = row.get('cvci_cate_mse_mean', float('nan'))
            lam = row.get('lambda_opt_mean', float('nan'))

            # Check if CVCI is best
            others = [exp_c, obs_c, pool_c]
            valid_others = [x for x in others if x == x]
            is_best = cvci_c == cvci_c and valid_others and cvci_c <= min(valid_others)

            print(f"{pv:>10.3f} | {exp_c:>10.4f} | {obs_c:>10.4f} | "
                  f"{pool_c:>10.4f} | {cvci_c:>10.4f} | {lam:>6.3f} | "
                  f"{'  YES' if is_best else '  no':>10}")


def create_combined_plot(found, results_dir):
    """Create a 3×3 grid of CATE MSE plots (axes × CATE functions)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    axes_order = ['epsilon', 'nobs', 'confounding']
    cates_order = ['constant', 'step', 'nonlinear']
    x_labels = {
        'epsilon': 'Treatment Effect Bias',
        'nobs': 'N_obs',
        'confounding': 'Confounding Strength',
    }
    param_keys = {
        'epsilon': 'epsilon',
        'nobs': 'n_obs',
        'confounding': 'confounding',
    }

    methods = [
        ('exp_only', 'Exp-only CF', '#2ca02c', '--', 'o'),
        ('obs_only', 'Obs-only CF', '#d62728', '--', 's'),
        ('pooled', 'Pooled CF', '#9467bd', ':', '^'),
        ('cvci', 'CVCI-CF', '#1f77b4', '-', 'D'),
    ]

    fig, axgrid = plt.subplots(3, 3, figsize=(16, 14))

    for row_idx, axis in enumerate(axes_order):
        for col_idx, cate in enumerate(cates_order):
            ax = axgrid[row_idx][col_idx]

            key = (axis, cate)
            if key not in found:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title(f'{cate}', fontsize=11)
                continue

            summary_path = os.path.join(found[key], 'summary.json')
            if not os.path.exists(summary_path):
                ax.text(0.5, 0.5, 'No summary', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                continue

            summary = load_json(summary_path)
            x_key = param_keys[axis]
            x_vals = [row[x_key] for row in summary]

            for method, label, color, ls, marker in methods:
                y = [row.get(f'{method}_cate_mse_mean', float('nan'))
                     for row in summary]
                se = [row.get(f'{method}_cate_mse_se', 0)
                      for row in summary]
                ax.errorbar(x_vals, y, yerr=se, label=label,
                            color=color, linestyle=ls, marker=marker,
                            markersize=4, capsize=2, linewidth=1.2)

            ax.grid(alpha=0.3)

            # Labels
            if row_idx == 2:
                ax.set_xlabel(x_labels[axis], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('CATE MSE', fontsize=10)
            if row_idx == 0:
                ax.set_title(f'{cate}', fontsize=12, fontweight='bold')

            # Row label on the right
            if col_idx == 2:
                ax2 = ax.twinx()
                ax2.set_ylabel(f'Varying {axis}', fontsize=11,
                               rotation=270, labelpad=15)
                ax2.set_yticks([])

            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=7, loc='upper left')

    plt.suptitle('CVCI-CF: CATE MSE Across Simulation Axes and CATE Functions',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = os.path.join(results_dir, 'combined_cate_mse.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Combined plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect and summarize CVCI-CF results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--plot', action='store_true', help='Create combined plot')
    args = parser.parse_args()

    found = check_results(args.results_dir)
    if found:
        print_combined_summary(found)
    if args.plot and found:
        create_combined_plot(found, args.results_dir)


if __name__ == '__main__':
    main()
