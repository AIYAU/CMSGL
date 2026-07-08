from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASET_NAMES = {1: "Trento", 2: "Houston", 3: "MUUFL"}
METRICS = ["unknown_accuracy", "oa", "aa", "kappa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate best TTR operating points across datasets and seeds.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_prefix", default="three_dataset_ttr_summary")
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def to_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "-1")))
    except ValueError:
        return -1


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    by_dataset_seed: dict[tuple[int, int], dict[str, str]] = {}
    details: list[dict[str, object]] = []

    for input_arg in args.inputs:
        for path in sorted(Path().glob(input_arg) if any(ch in input_arg for ch in "*?[") else [Path(input_arg)]):
            if not path.exists():
                continue
            for row in read_rows(path):
                data_id = to_int(row, "data")
                seed = to_int(row, "seed")
                if data_id not in DATASET_NAMES or seed < 0:
                    continue
                key = (data_id, seed)
                current = by_dataset_seed.get(key)
                if current is None or to_float(row, "min_margin_vs_hyliosr") > to_float(current, "min_margin_vs_hyliosr"):
                    by_dataset_seed[key] = row

    for (data_id, seed), row in sorted(by_dataset_seed.items()):
        record: dict[str, object] = {
            "dataset": DATASET_NAMES[data_id],
            "data": data_id,
            "seed": seed,
            "beat_hyliosr_all4": to_int(row, "beat_hyliosr_all4"),
            "min_margin_vs_hyliosr": to_float(row, "min_margin_vs_hyliosr"),
            "mode": row.get("mode", ""),
            "threshold_mode": row.get("threshold_mode", ""),
            "threshold_quantile": to_float(row, "threshold_quantile"),
            "run_dir": row.get("run_dir", ""),
        }
        for metric in METRICS:
            record[metric] = to_float(row, metric)
            record[f"{metric}_margin"] = to_float(row, f"{metric}_margin")
        details.append(record)

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in details:
        grouped[int(row["seed"])].append(row)

    seed_rows: list[dict[str, object]] = []
    for seed, rows in sorted(grouped.items()):
        if len(rows) < 3:
            continue
        record: dict[str, object] = {
            "seed": seed,
            "num_datasets": len(rows),
            "beat_all_datasets_all4": int(all(int(row["beat_hyliosr_all4"]) == 1 for row in rows)),
            "min_margin_across_datasets": min(float(row["min_margin_vs_hyliosr"]) for row in rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            margins = [float(row[f"{metric}_margin"]) for row in rows]
            record[f"{metric}_mean"] = sum(values) / len(values)
            record[f"{metric}_min"] = min(values)
            record[f"{metric}_margin_mean"] = sum(margins) / len(margins)
            record[f"{metric}_margin_min"] = min(margins)
        for data_id, name in DATASET_NAMES.items():
            match = next((row for row in rows if int(row["data"]) == data_id), None)
            if match is None:
                continue
            record[f"{name}_beat"] = int(match["beat_hyliosr_all4"])
            record[f"{name}_unknown"] = float(match["unknown_accuracy"])
            record[f"{name}_oa"] = float(match["oa"])
            record[f"{name}_aa"] = float(match["aa"])
            record[f"{name}_kappa"] = float(match["kappa"])
        seed_rows.append(record)

    seed_rows.sort(
        key=lambda row: (
            int(row["beat_all_datasets_all4"]),
            float(row["min_margin_across_datasets"]),
            float(row["kappa_mean"]),
            float(row["oa_mean"]),
        ),
        reverse=True,
    )

    detail_fields = [
        "dataset",
        "data",
        "seed",
        "beat_hyliosr_all4",
        "unknown_accuracy",
        "oa",
        "aa",
        "kappa",
        "unknown_accuracy_margin",
        "oa_margin",
        "aa_margin",
        "kappa_margin",
        "min_margin_vs_hyliosr",
        "mode",
        "threshold_mode",
        "threshold_quantile",
        "run_dir",
    ]
    seed_fields = [
        "seed",
        "num_datasets",
        "beat_all_datasets_all4",
        "min_margin_across_datasets",
        "unknown_accuracy_mean",
        "oa_mean",
        "aa_mean",
        "kappa_mean",
        "unknown_accuracy_min",
        "oa_min",
        "aa_min",
        "kappa_min",
        "unknown_accuracy_margin_min",
        "oa_margin_min",
        "aa_margin_min",
        "kappa_margin_min",
        "Trento_beat",
        "Trento_unknown",
        "Trento_oa",
        "Trento_aa",
        "Trento_kappa",
        "Houston_beat",
        "Houston_unknown",
        "Houston_oa",
        "Houston_aa",
        "Houston_kappa",
        "MUUFL_beat",
        "MUUFL_unknown",
        "MUUFL_oa",
        "MUUFL_aa",
        "MUUFL_kappa",
    ]

    detail_path = output_dir / f"{args.output_prefix}_details.csv"
    seed_path = output_dir / f"{args.output_prefix}_by_seed.csv"
    report_path = output_dir / f"{args.output_prefix}.md"
    write_csv(detail_path, details, detail_fields)
    write_csv(seed_path, seed_rows, seed_fields)

    lines = [
        "# Three-Dataset TTR Summary",
        "",
        f"Dataset-seed records: {len(details)}",
        f"Complete three-dataset seeds: {len(seed_rows)}",
        "",
        "| seed | all datasets beat | min margin | Unknown mean | OA mean | AA mean | Kappa mean | Trento | Houston | MUUFL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    "yes" if int(row["beat_all_datasets_all4"]) else "no",
                    format_metric(row["min_margin_across_datasets"]),
                    format_metric(row["unknown_accuracy_mean"]),
                    format_metric(row["oa_mean"]),
                    format_metric(row["aa_mean"]),
                    format_metric(row["kappa_mean"]),
                    f"{int(row.get('Trento_beat', 0))}",
                    f"{int(row.get('Houston_beat', 0))}",
                    f"{int(row.get('MUUFL_beat', 0))}",
                ]
            )
            + " |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {detail_path}")
    print(f"Wrote {seed_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
