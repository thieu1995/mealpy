#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:41, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from pandas import DataFrame
from mealpy.evolutionary_based.DE import BaseDE
from examples.setting_function import func_paras, func_names, problem_size
from os import getcwd, path, makedirs

model_name = "DE"
num_runs = 5
PATH_RESULTS = "history/results/"
check_dir1 = getcwd() + "/" + PATH_RESULTS
if not path.exists(check_dir1):
    makedirs(check_dir1)

## Setting parameters
epoch = 10
pop_size = 50
wf = 0.8
cr = 0.9

## Run model
best_fit_full = {}
best_fit_columns = []

error_full = {}
error_columns = []
for id_paras in range(len(func_paras)):
    md = BaseDE(func_paras[id_paras], epoch, pop_size, wf, cr)
    _, best_fit, list_loss = md._train__()

    error_full[func_names[id_paras]] = list_loss
    error_columns.append(func_names[id_paras])

    best_fit_full[func_names[id_paras]] = [best_fit]
    best_fit_columns.append(func_names[id_paras])

df_err = DataFrame(error_full, columns=error_columns)
df_err.to_csv(PATH_RESULTS + str(problem_size) + "D_" + model_name + "_error.csv", header=True, index=False)

df_fit = DataFrame(best_fit_full, columns=best_fit_columns)
df_fit.to_csv(PATH_RESULTS + str(problem_size) + "D_" + model_name + "_best_fit.csv", header=True, index=False)

