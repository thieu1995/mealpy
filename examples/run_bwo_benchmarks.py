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
from mealpy import FloatVar, BWO, __version__ as mealpy_version
from bwo_functions import get_bwo_functions


def _safe_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "function"


def _function_id(name: str) -> int:
    match = re.match(r"F(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse function id from name: {name}")
    return int(match.group(1))


def _build_table3_settings():
    default = [(10, 100, 500), (20, 150, 1000), (50, 200, 1500)]
    settings = {fid: list(default) for fid in range(1, 46)}
    settings[26] = [(2, 100, 500), (2, 150, 1000), (2, 500, 2000)]
    settings[27] = [(2, 100, 500), (2, 200, 1500), (2, 500, 1500)]
    settings[33] = [(2, 100, 500), (2, 200, 1500), (2, 500, 2000)]
    settings[34] = [(1, 100, 500), (1, 200, 1500), (1, 500, 1500)]
    settings[36] = [(2, 100, 500), (2, 200, 1500), (2, 500, 2000)]
    settings[37] = [(2, 100, 500), (2, 200, 1500), (2, 500, 1500)]
    settings[38] = [(2, 100, 500), (2, 200, 1500), (2, 500, 2000)]
    settings[39] = [(2, 100, 500), (2, 200, 1500), (2, 500, 2000)]
    settings[40] = [(2, 100, 500), (2, 200, 1000), (2, 500, 2000)]
    return settings


def _index_functions(n_dims: int, include_composites: bool) -> dict:
    funcs = get_bwo_functions(n_dims, include_composites=include_composites)
    return {_function_id(func.name): func for func in funcs}


def _build_problem(func, log_to, log_file):
    problem = {
        "bounds": FloatVar(lb=func.lb, ub=func.ub, name="x"),
        "minmax": "min",
        "obj_func": func.func,
        "name": func.name,
        "log_to": log_to,
    }
    if log_to == "file":
        problem["log_file"] = log_file
    return problem


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run BWO on F1-F45 (and optional composites) and save comparison-ready results.")
    parser.add_argument("--dims", type=int, default=50, help="Dimension for scalable functions.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs per run.")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size.")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent runs.")
    parser.add_argument("--seed", type=int, default=10, help="Base seed; run seed = seed + run index.")
    parser.add_argument("--pp", type=float, default=0.6, help="Procreating rate.")
    parser.add_argument("--cr", type=float, default=0.44, help="Cannibalism rate.")
    parser.add_argument("--pm", type=float, default=0.4, help="Mutation rate.")
    parser.add_argument("--log-to", choices=["console", "file", "none"], default="none", help="Mealpy logging target.")
    parser.add_argument("--output", default="examples/results/bwo", help="Output directory for results.")
    parser.add_argument("--tag", default="", help="Optional tag to create a unique subfolder.")
    parser.add_argument("--no-table3", action="store_true", help="Disable Table 3 settings and use global dims/pop/epochs.")
    parser.add_argument("--no-composites", action="store_true", help="Skip CEC2005 composite functions F46-F51.")
    parser.add_argument("--composite-pop-size", type=int, default=100, help="Population size for composite functions.")
    parser.add_argument("--composite-epochs", type=int, default=500, help="Epochs for composite functions.")
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

    use_table3 = not args.no_table3
    include_composites = not args.no_composites
    metadata = {
        "algorithm": "BWO",
        "bwo_params": {"pp": args.pp, "cr": args.cr, "pm": args.pm},
        "global_dims": args.dims,
        "global_epochs": args.epochs,
        "global_pop_size": args.pop_size,
        "runs": args.runs,
        "seed": args.seed,
        "table3_enabled": use_table3,
        "include_composites": include_composites,
        "composite_pop_size": args.composite_pop_size,
        "composite_epochs": args.composite_epochs,
        "mealpy_version": mealpy_version,
        "numpy_version": np.__version__,
        "timestamp": run_tag,
    }
    if use_table3:
        metadata["table3_settings"] = _build_table3_settings()
        metadata["table3_notes"] = "F36/F39/F40 use 2D to match function definitions and reported minima; Table 3 lists Nvar=1."
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")

    summary_rows = []
    run_rows = []

    if use_table3:
        table3_settings = _build_table3_settings()
        dims_required = sorted({dim for settings in table3_settings.values() for dim, _, _ in settings})
        if include_composites and 10 not in dims_required:
            dims_required.append(10)
        funcs_by_dim = {
            dim: _index_functions(dim, include_composites=include_composites and dim == 10)
            for dim in dims_required
        }
        cases = []
        for fid in range(1, 46):
            for dim, pop_size, epochs in table3_settings[fid]:
                cases.append((fid, dim, pop_size, epochs))
        if include_composites:
            for fid in range(46, 52):
                cases.append((fid, 10, args.composite_pop_size, args.composite_epochs))
    else:
        functions = get_bwo_functions(args.dims, include_composites=include_composites)
        cases = [(_function_id(func.name), func.n_dims, args.pop_size, args.epochs) for func in functions]
        funcs_by_dim = {}
        for func in functions:
            fid = _function_id(func.name)
            funcs_by_dim.setdefault(func.n_dims, {})[fid] = func

    for fid, dims, pop_size, epochs in cases:
        func = funcs_by_dim[dims].get(fid)
        if func is None:
            raise KeyError(f"Missing function F{fid} for dimension {dims}.")
        label = f"{func.name} (D={dims}, P={pop_size}, E={epochs})"
        best_list = []
        runtime_list = []
        histories = []

        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            np.random.seed(seed)

            problem = _build_problem(func, log_to, log_file)
            model = BWO.OriginalBWO(epoch=epochs, pop_size=pop_size, pp=args.pp, cr=args.cr, pm=args.pm)
            start = time.perf_counter()
            best = model.solve(problem, seed=seed)
            runtime = time.perf_counter() - start

            best_fit = float(best.target.fitness)
            best_list.append(best_fit)
            runtime_list.append(runtime)
            histories.append(model.history.list_global_best_fit)

            run_rows.append({
                "function_id": fid,
                "function": func.name,
                "dims": dims,
                "pop_size": pop_size,
                "epochs": epochs,
                "run": run_idx + 1,
                "seed": seed,
                "best_fitness": best_fit,
                "runtime_sec": runtime,
            })

        best_arr = np.asarray(best_list, dtype=float)
        runtime_arr = np.asarray(runtime_list, dtype=float)
        summary_rows.append({
            "function_id": fid,
            "function": func.name,
            "dims": dims,
            "pop_size": pop_size,
            "epochs": epochs,
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
        tag = f"{safe_name}_D{dims}_P{pop_size}_E{epochs}"
        convergence_path = curves_dir / f"{tag}_avg.csv"
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
            history_path = curves_dir / f"{tag}_runs.csv"
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

        print(f"{label} -> best: {np.min(best_arr):.6g}, mean: {np.mean(best_arr):.6g}")

    _write_csv(out_dir / "summary.csv", [
        "function_id", "function", "dims", "pop_size", "epochs", "runs", "best", "mean", "std", "median", "worst",
        "runtime_mean_sec", "runtime_std_sec"
    ], summary_rows)
    _write_csv(out_dir / "runs.csv", [
        "function_id", "function", "dims", "pop_size", "epochs", "run", "seed", "best_fitness", "runtime_sec"
    ], run_rows)

    print(f"Results saved under: {out_dir}")


if __name__ == "__main__":
    main()
