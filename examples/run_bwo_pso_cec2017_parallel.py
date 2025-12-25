#!/usr/bin/env python
# Created by "Thieu" at 11:40, 20/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import concurrent.futures as cf
import multiprocessing as mp

import numpy as np
from mealpy import FloatVar, BWO, PSO
from opfunu.cec_based.cec2017 import F152017, F192017, F202017


POP_SIZE = 50
RUNS = 10
EPOCHS = 1000
DIMS = 30
SEED0 = 10

FUNCTIONS = {
    15: F152017,
    19: F192017,
    20: F202017,
}

ALGORITHMS = {
    "BWO": ("BWO", BWO.OriginalBWO, {"pp": 0.6, "cr": 0.44, "pm": 0.4}),
    "PSO": ("PSO", PSO.OriginalPSO, {}),
}


def _build_problem(func):
    return {
        "bounds": FloatVar(lb=func.lb, ub=func.ub, name="x"),
        "minmax": "min",
        "obj_func": func.evaluate,
        "name": func.name,
    }


def _run_case(algo_name, func_id):
    func_cls = FUNCTIONS[func_id]
    func = func_cls(ndim=DIMS)
    problem = _build_problem(func)

    best_list = []
    for run_idx in range(RUNS):
        seed = SEED0 + run_idx
        np.random.seed(seed)
        _, algo_cls, algo_kwargs = ALGORITHMS[algo_name]
        model = algo_cls(epoch=EPOCHS, pop_size=POP_SIZE, **algo_kwargs)
        best = model.solve(problem, seed=seed)
        best_list.append(float(best.target.fitness))

    arr = np.asarray(best_list, dtype=float)
    return {
        "algorithm": algo_name,
        "func_id": func_id,
        "best": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if RUNS > 1 else 0.0,
    }


def _format_value(value):
    return f"{value:.6g}"


def main():
    tasks = [(algo_name, func_id) for algo_name in ALGORITHMS for func_id in FUNCTIONS]

    ctx = mp.get_context("spawn")
    results = []
    with cf.ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = [executor.submit(_run_case, algo, func_id) for algo, func_id in tasks]
        for future in cf.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: (item["func_id"], item["algorithm"]))

    for func_id in sorted(FUNCTIONS):
        print(f"F{func_id} results (D={DIMS}, P={POP_SIZE}, E={EPOCHS}, runs={RUNS})")
        for algo_name in sorted(ALGORITHMS):
            row = next(r for r in results if r["func_id"] == func_id and r["algorithm"] == algo_name)
            print(f"{algo_name}: best={_format_value(row['best'])} "
                  f"mean={_format_value(row['mean'])} std={_format_value(row['std'])}")
        print("-" * 60)

    header = ["Algorithm"]
    for func_id in sorted(FUNCTIONS):
        header.extend([f"F{func_id}_best", f"F{func_id}_mean", f"F{func_id}_std"])

    print("Result matrix:")
    print("\t".join(header))
    for algo_name in sorted(ALGORITHMS):
        row = [algo_name]
        for func_id in sorted(FUNCTIONS):
            data = next(r for r in results if r["func_id"] == func_id and r["algorithm"] == algo_name)
            row.extend([_format_value(data["best"]), _format_value(data["mean"]), _format_value(data["std"])])
        print("\t".join(row))


if __name__ == "__main__":
    main()
