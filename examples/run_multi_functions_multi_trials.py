#!/usr/bin/env python
# Created by "Thieu" at 11:27, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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
func_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"]

PATH_ERROR = "history/error/" + model_name + "/"
PATH_BEST_FIT = "history/best_fit/"
check_dir1 = f"{getcwd()}/{PATH_ERROR}"
check_dir2 = f"{getcwd()}/{PATH_BEST_FIT}"
if not path.exists(check_dir1): makedirs(check_dir1)
if not path.exists(check_dir2): makedirs(check_dir2)

## Run model
best_fit_full = {}
best_fit_columns = []
for func_name in func_names:
    error_full = {}
    error_columns = []
    best_fit_list = []
    for id_trial in range(1, N_TRIALS+1):
        problem = {
            "fit_func": getattr(cec2014_nobias, func_name),
            "lb": LB,
            "ub": UB,
            "minmax": "min",
            "verbose": True,
        }
        model = BaseDE(problem, epoch=epoch, pop_size=pop_size, wf=wf, cr=cr, name=model_name, fit_name=func_name)
        _, best_fitness = model.solve()

        temp = f"trial_{id_trial}"
        error_full[temp] = model.history.list_global_best_fit
        error_columns.append(temp)
        best_fit_list.append(best_fitness)
    df = DataFrame(error_full, columns=error_columns)

    df.to_csv(f"{PATH_ERROR}{len(LB)}D_{model_name}_{func_name}_error.csv", header=True, index=False)
    best_fit_full[func_name] = best_fit_list
    best_fit_columns.append(func_name)

df = DataFrame(best_fit_full, columns=best_fit_columns)
df.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_best_fit.csv", header=True, index=False)
