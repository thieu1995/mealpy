#!/usr/bin/env python
# Created by "Thieu" at 11:40, 20/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from opfunu.cec_based import cec2015
from mealpy import FloatVar, BWO, __version__ as mealpy_version

CEC2015_FUNCTIONS = [
    cec2015.F12015,
    cec2015.F22015,
    cec2015.F32015,
    cec2015.F42015,
    cec2015.F52015,
    cec2015.F62015,
    cec2015.F72015,
    cec2015.F82015,
    cec2015.F92015,
    cec2015.F102015,
    cec2015.F112015,
    cec2015.F122015,
    cec2015.F132015,
    cec2015.F142015,
    cec2015.F152015,
]


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "function"


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _build_problem(func, log_to, log_file):
    problem = {
        "bounds": FloatVar(lb=func.lb, ub=func.ub, name="x"),
        "minmax": "min",
        "obj_func": func.evaluate,
        "name": func.name,
        "log_to": log_to,
    }
    if log_to == "file":
        problem["log_file"] = log_file
    return problem


def _parse_args():
    parser = argparse.ArgumentParser(description="Run BWO on CEC2015 benchmarks and save comparison-ready results.")
    parser.add_argument("--dims", type=int, default=30, help="CEC2015 supported dimensions: 10 or 30.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs per run.")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size.")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent runs.")
    parser.add_argument("--seed", type=int, default=10, help="Base seed; run seed = seed + run index.")
    parser.add_argument("--pp", type=float, default=0.6, help="Procreating rate.")
    parser.add_argument("--cr", type=float, default=0.44, help="Cannibalism rate.")
    parser.add_argument("--pm", type=float, default=0.4, help="Mutation rate.")
    parser.add_argument("--log-to", choices=["console", "file", "none"], default="none", help="Mealpy logging target.")
    parser.add_argument("--output", default="examples/results/bwo_cec2015", help="Output directory for results.")
    parser.add_argument("--tag", default="", help="Optional tag to create a unique subfolder.")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias in CEC2015 functions.")
    parser.add_argument("--save-history", action="store_true", help="Save per-run convergence data.")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_tag = args.tag.strip() or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / f"bwo_{run_tag}"
    curves_dir = out_dir / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    log_to = None if args.log_to == "none" else args.log_to
    log_file = str(out_dir / "mealpy.log") if log_to == "file" else None

    metadata = {
        "algorithm": "BWO",
        "bwo_params": {"pp": args.pp, "cr": args.cr, "pm": args.pm},
        "dims": args.dims,
        "epochs": args.epochs,
        "pop_size": args.pop_size,
        "runs": args.runs,
        "seed": args.seed,
        "use_bias": not args.no_bias,
        "mealpy_version": mealpy_version,
        "numpy_version": np.__version__,
        "timestamp": run_tag,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")

    if args.no_bias:
        functions = [cls(args.dims, f_bias=0.0) for cls in CEC2015_FUNCTIONS]
    else:
        functions = [cls(args.dims) for cls in CEC2015_FUNCTIONS]

    summary_rows = []
    run_rows = []

    for func in functions:
        best_list = []
        runtime_list = []
        histories = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            np.random.seed(seed)

            problem = _build_problem(func, log_to, log_file)
            model = BWO.OriginalBWO(epoch=args.epochs, pop_size=args.pop_size, pp=args.pp, cr=args.cr, pm=args.pm)
            start = time.perf_counter()
            best = model.solve(problem, seed=seed)
            runtime = time.perf_counter() - start

            best_fit = float(best.target.fitness)
            best_list.append(best_fit)
            runtime_list.append(runtime)
            histories.append(model.history.list_global_best_fit)

            run_rows.append({
                "function": func.name,
                "dims": args.dims,
                "run": run_idx + 1,
                "seed": seed,
                "best_fitness": best_fit,
                "runtime_sec": runtime,
            })

        best_arr = np.asarray(best_list, dtype=float)
        runtime_arr = np.asarray(runtime_list, dtype=float)
        summary_rows.append({
            "function": func.name,
            "dims": args.dims,
            "runs": args.runs,
            "best": float(np.min(best_arr)),
            "mean": float(np.mean(best_arr)),
            "std": float(np.std(best_arr, ddof=1)) if args.runs > 1 else 0.0,
            "median": float(np.median(best_arr)),
            "worst": float(np.max(best_arr)),
            "runtime_mean_sec": float(np.mean(runtime_arr)),
            "runtime_std_sec": float(np.std(runtime_arr, ddof=1)) if args.runs > 1 else 0.0,
        })

        min_len = min(len(h) for h in histories)
        history_arr = np.array([h[:min_len] for h in histories], dtype=float)
        safe_name = _safe_slug(func.name)
        convergence_path = curves_dir / f"{safe_name}_avg.csv"
        with convergence_path.open("w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "mean_best", "std_best", "min_best", "max_best"])
            writer.writeheader()
            for epoch_idx in range(min_len):
                writer.writerow({
                    "epoch": epoch_idx,
                    "mean_best": float(np.mean(history_arr[:, epoch_idx])),
                    "std_best": float(np.std(history_arr[:, epoch_idx], ddof=1)) if args.runs > 1 else 0.0,
                    "min_best": float(np.min(history_arr[:, epoch_idx])),
                    "max_best": float(np.max(history_arr[:, epoch_idx])),
                })

        if args.save_history:
            history_path = curves_dir / f"{safe_name}_runs.csv"
            with history_path.open("w", newline="", encoding="ascii") as f:
                writer = csv.DictWriter(f, fieldnames=["run", "epoch", "best_fitness"])
                writer.writeheader()
                for run_idx, history in enumerate(histories, start=1):
                    for epoch_idx, value in enumerate(history[:min_len]):
                        writer.writerow({
                            "run": run_idx,
                            "epoch": epoch_idx,
                            "best_fitness": float(value),
                        })

        print(f"{func.name} -> best: {np.min(best_arr):.6g}, mean: {np.mean(best_arr):.6g}")

    _write_csv(out_dir / "summary.csv", [
        "function", "dims", "runs", "best", "mean", "std", "median", "worst", "runtime_mean_sec", "runtime_std_sec"
    ], summary_rows)
    _write_csv(out_dir / "runs.csv", [
        "function", "dims", "run", "seed", "best_fitness", "runtime_sec"
    ], run_rows)

    print(f"Results saved under: {out_dir}")


if __name__ == "__main__":
    main()
