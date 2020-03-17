#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:28, 16/03/2020                                                        %
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
PATH_ERROR = "history/error/" + model_name + "/"
PATH_BEST_FIT = "history/best_fit/"
check_dir1 = getcwd() + "/" + PATH_ERROR
check_dir2 = getcwd() + "/" + PATH_BEST_FIT
if not path.exists(check_dir1):
    makedirs(check_dir1)
if not path.exists(check_dir2):
    makedirs(check_dir2)

## Setting parameters
epoch = 10
pop_size = 50
wf = 0.8
cr = 0.9

## Run model
best_fit_full = {}
best_fit_columns = []
for id_paras in range(len(func_paras)):
    error_full = {}
    error_columns = []
    best_fit_list = []
    for id_runs in range(num_runs):
        md = BaseDE(func_paras[id_paras], epoch, pop_size, wf, cr)
        _, best_fit, list_loss = md._train__()
        temp = "time_" + str(id_runs+1)
        error_full[temp] = list_loss
        error_columns.append(temp)
        best_fit_list.append(best_fit)
    df = DataFrame(error_full, columns=error_columns)

    df.to_csv(PATH_ERROR + str(problem_size) + "D_" + model_name + "_" + func_names[id_paras] + "_error.csv", header=True, index=False)
    best_fit_full[func_names[id_paras]] = best_fit_list
    best_fit_columns.append(func_names[id_paras])

df = DataFrame(best_fit_full, columns=best_fit_columns)
df.to_csv(PATH_BEST_FIT + str(problem_size) + "D_" + model_name + "_best_fit.csv", header=True, index=False)

