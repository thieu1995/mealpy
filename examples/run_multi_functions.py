#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
from opfunu.cec_based import cec2005
from pandas import DataFrame
from mealpy import FloatVar, DE


PATH_RESULTS = "history/results/"
Path(PATH_RESULTS).mkdir(parents=True, exist_ok=True)

## Setting parameters
model_name = "DE"
ndims = 30
lb1 = [-100, ] * ndims
ub1 = [100, ] * ndims
epoch = 10
pop_size = 50
wf = 0.8
cr = 0.9

func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]


## Run model
best_fit_full = {}
best_fit_columns = []

error_full = {}
error_columns = []
for fname in func_names:
    func_name = f"{fname}2005"
    FF = getattr(cec2005, func_name)(ndims, f_bias=0)
    problem = {
        "obj_func": FF.evaluate,
        "bounds": FloatVar(lb=FF.lb, ub=FF.ub),
        "minmax": "min",
        "log_to": "console",
    }
    model = DE.OriginalDE(epoch, pop_size, wf, cr)
    best_agent = model.solve(problem)

    error_full[func_name] = model.history.list_global_best_fit
    error_columns.append(func_name)

    best_fit_full[func_name] = [best_agent.target.fitness]
    best_fit_columns.append(func_name)

df_err = DataFrame(error_full, columns=error_columns)
df_err.to_csv(f"{PATH_RESULTS}{len(lb1)}D_{model_name}_error.csv", header=True, index=False)

df_fit = DataFrame(best_fit_full, columns=best_fit_columns)
df_fit.to_csv(f"{PATH_RESULTS}{len(lb1)}D_{model_name}_best_fit.csv", header=True, index=False)
