import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DLL_PATHS = [
    path
    for path in os.environ.get("GEOCALIB_DLL_PATHS", "").split(os.pathsep)
    if path
]


def build_env():
    env = os.environ.copy()
    if DLL_PATHS:
        env["PATH"] = os.pathsep.join(DLL_PATHS + [env.get("PATH", "")])
    return env


def contains_frame_json(path):
    return any((child / "fused_inference_results.json").is_file() for child in path.iterdir() if child.is_dir())


def iter_v2x_pair_roots(root):
    root = Path(root)
    if not root.is_dir():
        return
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for scene_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for pair_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
                if contains_frame_json(pair_dir):
                    yield split_dir.name, scene_dir.name, pair_dir


def run_command(command, env, log_path=None):
    print(" ".join(str(part) for part in command), flush=True)
    if log_path is None:
        subprocess.run(command, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        subprocess.run(command, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def safe_name(value):
    return value.replace("\\", "_").replace("/", "_").replace(":", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", default="build-gtsam-viewer/Release/v2x_gtsam_viewer.exe")
    parser.add_argument("--dair-root", default="Dair-v2xdataset/output_cooperative_world_scenes")
    parser.add_argument("--v2x-root", default="v2xreal_gtsam_all_world_scenes")
    parser.add_argument("--out-root", default="eval_results/gtsam_full")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--dair-delay", type=int, default=0)
    parser.add_argument("--v2x-delay", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke-test limit. Use 0 for full evaluation.")
    parser.add_argument("--max-v2x-sequences", type=int, default=0, help="Smoke-test limit. Use 0 for all V2X-Real sequences.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip sequence CSV files that already exist.")
    parser.add_argument("--per-sequence-v2x", action="store_true", help="Launch one viewer process per V2X-Real pair root.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel V2X-Real sequence workers. Values >1 imply --per-sequence-v2x.")
    parser.add_argument("--match-dist", type=float, default=None)
    parser.add_argument("--min-boxes", type=int, default=None)
    parser.add_argument("--min-center-only", type=int, default=None)
    parser.add_argument("--graph-min-candidates", type=int, default=None)
    parser.add_argument("--graph-consistency-gate", type=float, default=None)
    parser.add_argument("--graph-min-inlier-ratio", type=float, default=None)
    parser.add_argument("--scene-match-min-score", type=float, default=None)
    parser.add_argument("--yaw-adaptive-gate", type=float, default=None)
    parser.add_argument("--yaw-reject-gate", type=float, default=None)
    parser.add_argument("--size-ratio-gate", type=float, default=None)
    parser.add_argument("--max-scene-candidates", type=int, default=None)
    parser.add_argument("--max-frame-observations", type=int, default=None)
    parser.add_argument("--rematch-iterations", type=int, default=None)
    parser.add_argument("--max-mean-error", type=float, default=None)
    parser.add_argument("--max-correction-trans", type=float, default=None)
    parser.add_argument("--max-correction-yaw", type=float, default=None)
    parser.add_argument("--max-step-trans", type=float, default=None)
    parser.add_argument("--max-step-yaw", type=float, default=None)
    parser.add_argument("--icp-static", action="store_true", help="Enable static-background ICP geometry factor.")
    parser.add_argument("--icp-max-boxes", type=int, default=None, help="Only compute ICP when accepted boxes are at or below this count.")
    parser.add_argument("--icp-max-points", type=int, default=None, help="Maximum static points used by each ICP solve.")
    parser.add_argument("--icp-iterations", type=int, default=None, help="Number of static ICP iterations.")
    parser.add_argument("--icp-min-correspondences", type=int, default=None)
    parser.add_argument("--icp-grid", type=float, default=None)
    parser.add_argument("--icp-max-correspondence", type=float, default=None)
    parser.add_argument("--icp-max-rmse", type=float, default=None)
    parser.add_argument("--icp-max-trans", type=float, default=None)
    parser.add_argument("--icp-max-yaw", type=float, default=None)
    parser.add_argument("--icp-box-margin", type=float, default=None)
    parser.add_argument("--icp-trans-sigma", type=float, default=None)
    parser.add_argument("--icp-yaw-sigma", type=float, default=None)
    parser.add_argument("--supervised-prior-csv", default="", help="CSV exported by train/v2i_supervised_pose_prior.py.")
    parser.add_argument("--supervised-min-confidence", type=float, default=None)
    parser.add_argument("--supervised-max-sigma-trans", type=float, default=None)
    parser.add_argument("--supervised-max-sigma-yaw", type=float, default=None)
    parser.add_argument("--skip-dair", action="store_true")
    parser.add_argument("--skip-v2x", action="store_true")
    args = parser.parse_args()

    viewer = Path(args.viewer)
    if not viewer.is_file():
        raise SystemExit(f"Viewer not found: {viewer}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "logs"
    env = build_env()

    def append_optimizer_args(command):
        option_map = (
            ("match_dist", "--match-dist"),
            ("min_boxes", "--min-boxes"),
            ("min_center_only", "--min-center-only"),
            ("graph_min_candidates", "--graph-min-candidates"),
            ("graph_consistency_gate", "--graph-consistency-gate"),
            ("graph_min_inlier_ratio", "--graph-min-inlier-ratio"),
            ("scene_match_min_score", "--scene-match-min-score"),
            ("yaw_adaptive_gate", "--yaw-adaptive-gate"),
            ("yaw_reject_gate", "--yaw-reject-gate"),
            ("size_ratio_gate", "--size-ratio-gate"),
            ("max_scene_candidates", "--max-scene-candidates"),
            ("max_frame_observations", "--max-frame-observations"),
            ("rematch_iterations", "--rematch-iterations"),
            ("max_mean_error", "--max-mean-error"),
            ("max_correction_trans", "--max-correction-trans"),
            ("max_correction_yaw", "--max-correction-yaw"),
            ("max_step_trans", "--max-step-trans"),
            ("max_step_yaw", "--max-step-yaw"),
        )
        for attr, flag in option_map:
            value = getattr(args, attr)
            if value is not None:
                command += [flag, str(value)]
        return command

    def append_icp_args(command):
        if not args.icp_static:
            return command
        command += ["--icp-static"]
        option_map = (
            ("icp_max_boxes", "--icp-max-boxes"),
            ("icp_max_points", "--icp-max-points"),
            ("icp_iterations", "--icp-iterations"),
            ("icp_min_correspondences", "--icp-min-correspondences"),
            ("icp_grid", "--icp-grid"),
            ("icp_max_correspondence", "--icp-max-correspondence"),
            ("icp_max_rmse", "--icp-max-rmse"),
            ("icp_max_trans", "--icp-max-trans"),
            ("icp_max_yaw", "--icp-max-yaw"),
            ("icp_box_margin", "--icp-box-margin"),
            ("icp_trans_sigma", "--icp-trans-sigma"),
            ("icp_yaw_sigma", "--icp-yaw-sigma"),
        )
        for attr, flag in option_map:
            value = getattr(args, attr)
            if value is not None:
                command += [flag, str(value)]
        return command

    if not args.skip_dair:
        dair_root = Path(args.dair_root)
        if not dair_root.is_dir():
            raise SystemExit(f"DAIR root not found: {dair_root}")
        dair_metrics_path = out_root / "dair_v2x.csv"
        command = [
            str(viewer),
            "--root",
            str(dair_root),
            "--delay",
            str(args.dair_delay),
            "--window",
            str(args.window),
            "--headless",
            "--metrics-csv",
            str(dair_metrics_path),
        ]
        command = append_optimizer_args(command)
        if args.max_frames > 0:
            command += ["--max-frames", str(args.max_frames)]
        command = append_icp_args(command)
        if args.supervised_prior_csv:
            command += ["--supervised-prior-csv", str(args.supervised_prior_csv)]
            if args.supervised_min_confidence is not None:
                command += ["--supervised-min-confidence", str(args.supervised_min_confidence)]
            if args.supervised_max_sigma_trans is not None:
                command += ["--supervised-max-sigma-trans", str(args.supervised_max_sigma_trans)]
            if args.supervised_max_sigma_yaw is not None:
                command += ["--supervised-max-sigma-yaw", str(args.supervised_max_sigma_yaw)]
        if args.skip_existing and dair_metrics_path.is_file() and dair_metrics_path.stat().st_size > 0:
            print(f"skip existing: {dair_metrics_path}", flush=True)
        else:
            run_command(command, env, log_root / "dair_v2x.log")

    if not args.skip_v2x:
        v2x_root = Path(args.v2x_root)
        if not v2x_root.is_dir():
            raise SystemExit(f"V2X-Real root not found: {v2x_root}")
        pair_roots = list(iter_v2x_pair_roots(v2x_root))
        if not pair_roots:
            raise SystemExit(f"No V2X-Real pair roots found under {v2x_root}")

        if args.max_v2x_sequences > 0:
            pair_roots = pair_roots[: args.max_v2x_sequences]

        total_pairs = len(pair_roots)
        per_sequence_v2x = args.per_sequence_v2x or args.jobs > 1

        def append_supervised_args(command):
            if not args.supervised_prior_csv:
                return command
            command += ["--supervised-prior-csv", str(args.supervised_prior_csv)]
            if args.supervised_min_confidence is not None:
                command += ["--supervised-min-confidence", str(args.supervised_min_confidence)]
            if args.supervised_max_sigma_trans is not None:
                command += ["--supervised-max-sigma-trans", str(args.supervised_max_sigma_trans)]
            if args.supervised_max_sigma_yaw is not None:
                command += ["--supervised-max-sigma-yaw", str(args.supervised_max_sigma_yaw)]
            return command

        if not per_sequence_v2x:
            root_list_path = out_root / "v2x_real_roots.txt"
            root_list_path.write_text(
                "\n".join(str(pair_root) for _, _, pair_root in pair_roots) + "\n",
                encoding="utf-8",
            )
            metrics_path = out_root / "v2x_real.csv"
            if args.skip_existing and metrics_path.is_file() and metrics_path.stat().st_size > 0:
                print(f"skip existing: {metrics_path}", flush=True)
            else:
                command = [
                    str(viewer),
                    "--root-list",
                    str(root_list_path),
                    "--delay",
                    str(args.v2x_delay),
                    "--window",
                    str(args.window),
                    "--v2x-real",
                    "--headless",
                    "--metrics-csv",
                    str(metrics_path),
                ]
                if args.max_frames > 0:
                    command += ["--max-frames", str(args.max_frames)]
                command = append_optimizer_args(command)
                command = append_icp_args(command)
                command = append_supervised_args(command)
                run_command(command, env, log_root / "v2x_real.log")
        else:
            def run_pair(task):
                idx, split_name, scene_name, pair_root = task
                print(f"[V2X-Real {idx}/{total_pairs}] {split_name}/{scene_name}/{pair_root.name}", flush=True)
                metrics_path = out_root / "v2x_real" / split_name / f"{safe_name(scene_name)}__{pair_root.name}.csv"
                log_path = log_root / "v2x_real" / split_name / f"{safe_name(scene_name)}__{pair_root.name}.log"
                if args.skip_existing and metrics_path.is_file() and metrics_path.stat().st_size > 0:
                    print(f"skip existing: {metrics_path}", flush=True)
                    return
                command = [
                    str(viewer),
                    "--root",
                    str(pair_root),
                    "--delay",
                    str(args.v2x_delay),
                    "--window",
                    str(args.window),
                    "--v2x-real",
                    "--headless",
                    "--metrics-csv",
                    str(metrics_path),
                ]
                if args.max_frames > 0:
                    command += ["--max-frames", str(args.max_frames)]
                command = append_optimizer_args(command)
                command = append_icp_args(command)
                command = append_supervised_args(command)
                run_command(command, env, log_path)

            tasks = [
                (idx, split_name, scene_name, pair_root)
                for idx, (split_name, scene_name, pair_root) in enumerate(pair_roots, start=1)
            ]
            jobs = max(1, args.jobs)
            if jobs == 1:
                for task in tasks:
                    run_pair(task)
            else:
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    futures = [executor.submit(run_pair, task) for task in tasks]
                    for future in as_completed(futures):
                        future.result()

    summary_command = [
        os.environ.get("PYTHON", "python"),
        "train/summarize_gtsam_metrics.py",
        "--metrics-root",
        str(out_root),
    ]
    run_command(summary_command, env)


if __name__ == "__main__":
    main()
