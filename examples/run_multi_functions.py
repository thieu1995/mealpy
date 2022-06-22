#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_basic import cec2014_nobias
from pandas import DataFrame
from mealpy.evolutionary_based.DE import BaseDE
from os import getcwd, path, makedirs

PATH_RESULTS = "history/results/"
check_dir = f"{getcwd()}/{PATH_RESULTS}"
if not path.exists(check_dir):
    makedirs(check_dir)

## Setting parameters
model_name = "DE"
lb1 = [-100, ] * 30
ub1 = [100, ] * 30
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
for func_name in func_names:
    problem = {
        "fit_func": getattr(cec2014_nobias, func_name),
        "lb": lb1,
        "ub": ub1,
        "minmax": "min",
        "log_to": "console",
    }
    model = BaseDE(problem, epoch, pop_size, wf, cr, fit_name=func_name)
    _, best_fitness = model.solve()

    error_full[func_name] = model.history.list_global_best_fit
    error_columns.append(func_name)

    best_fit_full[func_name] = [best_fitness]
    best_fit_columns.append(func_name)

df_err = DataFrame(error_full, columns=error_columns)
df_err.to_csv(f"{PATH_RESULTS}{len(lb1)}D_{model_name}_error.csv", header=True, index=False)

df_fit = DataFrame(best_fit_full, columns=best_fit_columns)
df_fit.to_csv(f"{PATH_RESULTS}{len(lb1)}D_{model_name}_best_fit.csv", header=True, index=False)
