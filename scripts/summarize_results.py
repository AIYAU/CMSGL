from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    'known_accuracy', 'unknown_accuracy', 'macro_f1', 'auroc', 'oscr',
    'oa', 'aa', 'kappa', 'threshold',
    'test_known_open_score_mean', 'test_unknown_open_score_mean',
    'known_E_h_mean', 'known_E_l_mean', 'known_E_compat_mean', 'known_E_total_mean',
    'unknown_E_h_mean', 'unknown_E_l_mean', 'unknown_E_compat_mean', 'unknown_E_total_mean',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize CC-SGCL experiment outputs')
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for metrics_path in sorted(output_dir.rglob('metrics.json')):
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        metrics = payload['metrics']
        row = {'run_dir': str(metrics_path.parent)}
        for key in METRICS:
            row[key] = metrics.get(key, float('nan'))
        rows.append(row)

    summary_csv = output_dir / 'summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['run_dir'] + METRICS)
        writer.writeheader()
        writer.writerows(rows)

    summary_stats = []
    for key in METRICS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary_stats.append({
            'metric': key,
            'mean': float(np.nanmean(values)) if len(values) else float('nan'),
            'std': float(np.nanstd(values)) if len(values) else float('nan'),
            'mean_pm_std': f"{np.nanmean(values):.6f} +- {np.nanstd(values):.6f}" if len(values) else 'nan +- nan',
        })

    summary_mean_std_csv = output_dir / 'summary_mean_std.csv'
    with summary_mean_std_csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['metric', 'mean', 'std', 'mean_pm_std'])
        writer.writeheader()
        writer.writerows(summary_stats)

    print('Wrote:', summary_csv)
    print('Wrote:', summary_mean_std_csv)


if __name__ == '__main__':
    main()
