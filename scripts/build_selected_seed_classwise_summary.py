from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = [
    "mode",
    "threshold_mode",
    "threshold_quantile",
    "ttr_weight",
    "consistency_weight",
    "global_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate class-wise HyLiOSR-style metrics for selected seeds from source TTR CSV files."
    )
    parser.add_argument("--best_csv", required=True, help="Best-per-seed CSV produced by summarize_ttr_hyliosr_metrics.py.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seed list to aggregate.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--latex_output", default=None, help="Optional LaTeX rows output path.")
    parser.add_argument("--method_name", default="CoSGL")
    return parser.parse_args()


def nearly_equal(a: pd.Series, b: float) -> pd.Series:
    return (a.astype(float) - float(b)).abs() < 1e-8


def find_source_row(source_csv: Path, best_row: pd.Series) -> pd.Series:
    source = pd.read_csv(source_csv)
    metric_mask = (
        nearly_equal(source["unknown_accuracy"], best_row["unknown_accuracy"])
        & nearly_equal(source["oa"], best_row["oa"])
        & nearly_equal(source["aa"], best_row["aa"])
        & nearly_equal(source["kappa"], best_row["kappa"])
    )
    matched = source[metric_mask]
    if not matched.empty:
        return matched.iloc[0]

    mask = pd.Series(True, index=source.index)
    for column in MATCH_COLUMNS:
        if column not in source.columns or column not in best_row.index:
            continue
        if pd.api.types.is_numeric_dtype(source[column]):
            mask &= nearly_equal(source[column], best_row[column])
        else:
            mask &= source[column].astype(str) == str(best_row[column])
    matched = source[mask]
    if matched.empty:
        raise ValueError(f"No matching source row found in {source_csv}")
    return matched.iloc[0]


def format_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def main() -> None:
    args = parse_args()
    best_csv = Path(args.best_csv)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    best = pd.read_csv(best_csv)
    selected = best[best["seed"].isin(seeds)].copy()
    selected["seed"] = pd.Categorical(selected["seed"], categories=seeds, ordered=True)
    selected = selected.sort_values("seed")
    missing = sorted(set(seeds) - set(int(seed) for seed in selected["seed"]))
    if missing:
        raise ValueError(f"Missing seeds in {best_csv}: {missing}")

    rows = []
    for _, best_row in selected.iterrows():
        source_csv = Path(str(best_row["source_csv"]))
        source_row = find_source_row(source_csv, best_row)
        rows.append(source_row)
    source_rows = pd.DataFrame(rows)

    metric_columns = [
        column
        for column in source_rows.columns
        if column.startswith("class_") and column.endswith("_accuracy")
    ]
    metric_columns = sorted(metric_columns, key=lambda name: int(name.split("_")[1]))
    metric_columns.extend(["unknown_accuracy", "oa", "aa", "kappa"])

    out_rows = []
    for column in metric_columns:
        values = source_rows[column].astype(float)
        label = column
        if column.startswith("class_"):
            label = f"Class {int(column.split('_')[1])}"
        elif column == "unknown_accuracy":
            label = "Unknown"
        else:
            label = column.upper() if column != "kappa" else "Kappa"
        out_rows.append(
            {
                "label": label,
                "metric_column": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "latex": format_pm(float(values.mean()), float(values.std(ddof=1))),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output, index=False)

    if args.latex_output:
        latex_output = Path(args.latex_output)
        latex_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"% Aggregated seeds: {', '.join(map(str, seeds))}"]
        for row in out_rows:
            lines.append(f"{row['label']} & {args.method_name} & {row['latex']} \\\\")
        latex_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {output}")
    if args.latex_output:
        print(f"Wrote {args.latex_output}")


if __name__ == "__main__":
    main()
