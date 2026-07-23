#!/usr/bin/env python
# Created by "Thieu" at 11:27, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
from opfunu.cec_based import cec2005
from pandas import DataFrame
from mealpy import FloatVar, DE


model_name = "DE"
N_TRIALS = 3
N_DIMS = 30
verbose = True
epoch = 10
pop_size = 50
wf = 0.8
cr = 0.9
func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]

PATH_ERROR = "history1/error/" + model_name + "/"
PATH_BEST_FIT = "history1/best_fit/"
Path(PATH_ERROR).mkdir(parents=True, exist_ok=True)
Path(PATH_BEST_FIT).mkdir(parents=True, exist_ok=True)

## Run model
best_fit_full = {}
best_fit_columns = []
for func_name in func_names:
    error_full = {}
    best_fit_list = []
    for id_trial in range(1, N_TRIALS+1):
        fname = f"{func_name}2005"
        FF = getattr(cec2005, fname)(N_DIMS, f_bias=0)
        problem = {
            "obj_func": FF.evaluate,
            "bounds": FloatVar(lb=FF.lb, ub=FF.ub),
            "minmax": "min",
            "log_to": "console",
            "name": func_name
        }
        model = DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, name=model_name)
        best_agent = model.solve(problem)

        temp = f"trial_{id_trial}"
        error_full[temp] = model.history.list_global_best_fit
        best_fit_list.append(best_agent.target.fitness)
    df = DataFrame(error_full)

    df.to_csv(f"{PATH_ERROR}{N_DIMS}D_{model_name}_{func_name}_error.csv", header=True, index=False)
    best_fit_full[func_name] = best_fit_list
    best_fit_columns.append(func_name)

df = DataFrame(best_fit_full, columns=best_fit_columns)
df.to_csv(f"{PATH_BEST_FIT}/{N_DIMS}D_{model_name}_best_fit.csv", header=True, index=False)
