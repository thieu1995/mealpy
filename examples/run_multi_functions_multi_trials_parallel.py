#!/usr/bin/env python
# Created by "Thieu" at 11:37, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import concurrent.futures as parallel
from pathlib import Path
from opfunu.cec_based import cec2005
from pandas import DataFrame
from mealpy import FloatVar, DE


model_name = "DE"
N_TRIALS = 5
N_DIMS = 30
verbose = True
epoch = 100
pop_size = 50
wf = 0.8
cr = 0.9
func_names = ["F1", "F2", "F3"]

PATH_ERROR = "history/error/" + model_name + "/"
PATH_BEST_FIT = "history/best_fit/"
Path(PATH_ERROR).mkdir(parents=True, exist_ok=True)
Path(PATH_BEST_FIT).mkdir(parents=True, exist_ok=True)

def find_minimum(function_name):
    """
    We can run multiple functions at the same time.
    Each core (CPU) will handle a function, each function will run N_TRIALS times
    """
    print(f"Start running: {function_name}")
    error_full = {}
    error_columns = []
    best_fit_list = []
    for id_trial in range(1, N_TRIALS + 1):
        fname = f"{function_name}2005"
        FF = getattr(cec2005, fname)(N_DIMS, f_bias=0)
        problem = {
            "obj_func": FF.evaluate,
            "bounds": FloatVar(lb=FF.lb, ub=FF.ub),
            "minmax": "min",
            "log_to": "console",
            "name": function_name
        }
        model = DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, name=model_name)
        best_agent = model.solve(problem)

        temp = f"trial_{id_trial}"
        error_full[temp] = model.history.list_global_best_fit
        error_columns.append(temp)
        best_fit_list.append(best_agent.target.fitness)
    df = DataFrame(error_full, columns=error_columns)
    df.to_csv(f"{PATH_ERROR}{N_DIMS}D_{model_name}_{function_name}_error.csv", header=True, index=False)
    print(f"Finish function: {function_name}")

    return {
        "func_name": function_name,
        "best_fit_list": best_fit_list,
        "model_name": model_name
    }


if __name__ == '__main__':
    ## Run model
    best_fit_full = {}
    best_fit_columns = []

    with parallel.ProcessPoolExecutor() as executor:
        results = executor.map(find_minimum, func_names)

    for result in results:
        best_fit_full[result["func_name"]] = result["best_fit_list"]
        best_fit_columns.append(result["func_name"])

    df = DataFrame(best_fit_full, columns=best_fit_columns)
    df.to_csv(f"{PATH_BEST_FIT}/{N_DIMS}D_{model_name}_best_fit.csv", header=True, index=False)
