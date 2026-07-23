#!/usr/bin/env python
# Created by "Thieu" at 10:26, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
from pathlib import Path
from opfunu.cec_based import cec2005
from pandas import DataFrame
from mealpy import FloatVar, DE


PATH_RESULTS = "history/results/"
Path(PATH_RESULTS).mkdir(parents=True, exist_ok=True)

model_name = "DE"
N_DIMS = 30
func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]


def find_minimum(function_name, n_dims):
    print(f"Start running: {function_name}")
    fname = f"{function_name}2005"
    FF = getattr(cec2005, fname)(N_DIMS, f_bias=0)
    problem = {
        "obj_func": FF.evaluate,
        "bounds": FloatVar(lb=FF.lb, ub=FF.ub),
        "minmax": "min",
        "log_to": "console",
        "name": function_name
    }
    model = DE.OriginalDE(epoch=10, pop_size=50, wf=0.8, cr=0.9, name=model_name)
    best_agent = model.solve(problem)
    print(f"Finish function: {function_name}")

    return {
        "func_name": function_name,
        "best_fit": [best_agent.target.fitness],
        "error": model.history.list_global_best_fit
    }


if __name__ == '__main__':
    ## Run model
    best_fit_full = {}
    best_fit_columns = []
    error_full = {}
    error_columns = []

    with parallel.ProcessPoolExecutor() as executor:
        results = executor.map(partial(find_minimum, n_dims=N_DIMS), func_names)

    for result in results:
        error_full[result["func_name"]] = result["error"]
        error_columns.append(result["func_name"])
        best_fit_full[result["func_name"]] = result["best_fit"]
        best_fit_columns.append(result["func_name"])

    df_err = DataFrame(error_full, columns=error_columns)
    df_err.to_csv(f"{PATH_RESULTS}{N_DIMS}D_{model_name}_error.csv", header=True, index=False)

    df_fit = DataFrame(best_fit_full, columns=best_fit_columns)
    df_fit.to_csv(f"{PATH_RESULTS}{N_DIMS}D_{model_name}_best_fit.csv", header=True, index=False)
