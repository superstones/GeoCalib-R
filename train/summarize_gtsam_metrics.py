import argparse
import csv
import json
import math
from pathlib import Path


SUMMARY_COLUMNS = [
    "quality",
    "delta_trans",
    "yaw_deg_abs",
    "graph_error",
    "mean_error",
    "init_center_error",
    "opt_center_error",
    "center_error_delta",
    "init_planar_center_error",
    "opt_planar_center_error",
    "planar_center_error_delta",
    "init_yaw_error_deg",
    "opt_yaw_error_deg",
    "yaw_error_delta",
    "icp_rmse",
    "icp_pairs",
    "rematch",
    "accepted",
]


def to_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def percentile(values, pct):
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def stats(values):
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "rmse": math.nan,
            "lt_0_5": math.nan,
            "lt_0_8": math.nan,
            "lt_1": math.nan,
            "lt_2": math.nan,
            "lt_3": math.nan,
            "gt_2_count": 0,
            "gt_3_count": 0,
            "gt_5_count": 0,
        }
    count = len(clean)
    return {
        "count": count,
        "mean": sum(clean) / count,
        "median": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "rmse": math.sqrt(sum(v * v for v in clean) / count),
        "lt_0_5": sum(v < 0.5 for v in clean) / count,
        "lt_0_8": sum(v < 0.8 for v in clean) / count,
        "lt_1": sum(v < 1.0 for v in clean) / count,
        "lt_2": sum(v < 2.0 for v in clean) / count,
        "lt_3": sum(v < 3.0 for v in clean) / count,
        "gt_2_count": sum(v > 2.0 for v in clean),
        "gt_3_count": sum(v > 3.0 for v in clean),
        "gt_5_count": sum(v > 5.0 for v in clean),
    }


def read_rows(csv_paths):
    rows = []
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["_csv"] = str(csv_path)
                rows.append(row)
    return rows


def collect_csv_paths(root):
    csv_paths = sorted(root.rglob("*.csv"))
    has_per_sequence_v2x = False
    for csv_path in csv_paths:
        try:
            rel_parts = csv_path.relative_to(root).parts
        except ValueError:
            continue
        if rel_parts and rel_parts[0] == "v2x_real":
            has_per_sequence_v2x = True
            break

    if has_per_sequence_v2x:
        root_v2x_csv = root / "v2x_real.csv"
        csv_paths = [csv_path for csv_path in csv_paths if csv_path != root_v2x_csv]
    return csv_paths


def summarize_rows(rows):
    datasets = sorted({row.get("dataset", "unknown") for row in rows})
    result = {}
    for dataset in datasets + ["all"]:
        subset = rows if dataset == "all" else [row for row in rows if row.get("dataset") == dataset]
        if not subset:
            continue

        rejected = [row for row in subset if row.get("rejected") == "1"]
        icp_rows = [row for row in subset if row.get("icp_used") == "1"]
        with_matches = [row for row in subset if int(row.get("accepted", "0") or 0) > 0]
        raw_pairs = sum(int(row.get("raw_pairs", "0") or 0) for row in subset)
        accepted = sum(int(row.get("accepted", "0") or 0) for row in subset)

        metrics = {}
        for col in SUMMARY_COLUMNS:
            if col == "yaw_deg_abs":
                values = [abs(to_float(row.get("yaw_deg"))) for row in subset]
            else:
                values = [to_float(row.get(col)) for row in subset]
            metrics[col] = stats(values)

        result[dataset] = {
            "frames": len(subset),
            "frames_with_matches": len(with_matches),
            "match_frame_rate": len(with_matches) / len(subset),
            "raw_pairs": raw_pairs,
            "accepted_pairs": accepted,
            "raw_pairs_per_frame": raw_pairs / len(subset),
            "accepted_pairs_per_frame": accepted / len(subset),
            "accepted_to_raw_ratio": accepted / raw_pairs if raw_pairs else math.nan,
            "pair_acceptance_rate": accepted / raw_pairs if raw_pairs else math.nan,
            "rejected_frames": len(rejected),
            "reject_rate": len(rejected) / len(subset),
            "icp_frames": len(icp_rows),
            "icp_usage_rate": len(icp_rows) / len(subset),
            "metrics": metrics,
        }
    return result


def write_markdown(summary, path):
    lines = [
        "# GTSAM Joint Optimization Evaluation",
        "",
        "Metrics follow the pose-graph/calibration style used by the reference papers: translation correction, yaw correction, object alignment error before/after optimization, matching coverage, and hard-case rejection rate.",
        "",
        "| dataset | frames | match frame rate | accepted/raw | reject rate | icp use/rmse | center err init -> opt (m) | planar err init -> opt (m) | yaw err init -> opt (deg) | opt RMSE/p95 (m) | planar RMSE/p95 (m) | opt <0.5/<1/<2m | planar <0.5/<1/<2m | opt >2/>3/>5m | |delta t| mean/p95 (m) | |yaw| mean/p95 (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in summary.items():
        metrics = item["metrics"]
        center_init = metrics["init_center_error"]["mean"]
        center_opt = metrics["opt_center_error"]["mean"]
        center_opt_stats = metrics["opt_center_error"]
        planar_init = metrics["init_planar_center_error"]["mean"]
        planar_opt = metrics["opt_planar_center_error"]["mean"]
        planar_opt_stats = metrics["opt_planar_center_error"]
        yaw_init = metrics["init_yaw_error_deg"]["mean"]
        yaw_opt = metrics["opt_yaw_error_deg"]["mean"]
        trans_mean = metrics["delta_trans"]["mean"]
        trans_p95 = metrics["delta_trans"]["p95"]
        yaw_mean = metrics["yaw_deg_abs"]["mean"]
        yaw_p95 = metrics["yaw_deg_abs"]["p95"]
        icp_rmse = metrics["icp_rmse"]["mean"]
        lines.append(
            f"| {dataset} | {item['frames']} | {item['match_frame_rate']:.3f} | "
            f"{item['accepted_to_raw_ratio']:.3f} | {item['reject_rate']:.3f} | "
            f"{item['icp_usage_rate']:.3f}/{icp_rmse:.3f} | "
            f"{center_init:.3f} -> {center_opt:.3f} | {planar_init:.3f} -> {planar_opt:.3f} | "
            f"{yaw_init:.3f} -> {yaw_opt:.3f} | "
            f"{center_opt_stats['rmse']:.3f}/{center_opt_stats['p95']:.3f} | "
            f"{planar_opt_stats['rmse']:.3f}/{planar_opt_stats['p95']:.3f} | "
            f"{center_opt_stats['lt_0_5']:.1%}/{center_opt_stats['lt_1']:.1%}/{center_opt_stats['lt_2']:.1%} | "
            f"{planar_opt_stats['lt_0_5']:.1%}/{planar_opt_stats['lt_1']:.1%}/{planar_opt_stats['lt_2']:.1%} | "
            f"{center_opt_stats['gt_2_count']}/{center_opt_stats['gt_3_count']}/{center_opt_stats['gt_5_count']} | "
            f"{trans_mean:.3f}/{trans_p95:.3f} | {yaw_mean:.3f}/{yaw_p95:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", default="eval_results/gtsam_full")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    root = Path(args.metrics_root)
    csv_paths = collect_csv_paths(root)
    if not csv_paths:
        raise SystemExit(f"No metrics csv files found under {root}")

    rows = read_rows(csv_paths)
    summary = summarize_rows(rows)

    out_json = Path(args.out_json) if args.out_json else root / "summary.json"
    out_md = Path(args.out_md) if args.out_md else root / "summary.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Read {len(csv_paths)} csv files, {len(rows)} frames")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
