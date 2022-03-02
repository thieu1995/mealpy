#!/usr/bin/env python
# Created by "Thieu" at 11:37, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import concurrent.futures as parallel
from opfunu.cec_basic import cec2014_nobias
from pandas import DataFrame
from mealpy.evolutionary_based.DE import BaseDE
from os import getcwd, path, makedirs


model_name = "DE"
N_TRIALS = 5
LB = [-100, ] * 15
UB = [100, ] * 15
verbose = True
epoch = 100
pop_size = 50
wf = 0.8
cr = 0.9
func_names = ["F1", "F2", "F3"]

PATH_ERROR = "history/error/" + model_name + "/"
PATH_BEST_FIT = "history/best_fit/"
check_dir1 = f"{getcwd()}/{PATH_ERROR}"
check_dir2 = f"{getcwd()}/{PATH_BEST_FIT}"
if not path.exists(check_dir1): makedirs(check_dir1)
if not path.exists(check_dir2): makedirs(check_dir2)


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
        problem = {
            "fit_func": getattr(cec2014_nobias, function_name),
            "lb": LB,
            "ub": UB,
            "minmax": "min",
            "verbose": True,
        }
        model = BaseDE(problem, epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, name=model_name, fit_name=function_name)
        _, best_fitness = model.solve()

        temp = f"trial_{id_trial}"
        error_full[temp] = model.history.list_global_best_fit
        error_columns.append(temp)
        best_fit_list.append(best_fitness)
    df = DataFrame(error_full, columns=error_columns)
    df.to_csv(f"{PATH_ERROR}{len(LB)}D_{model_name}_{function_name}_error.csv", header=True, index=False)
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
    df.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_best_fit.csv", header=True, index=False)
