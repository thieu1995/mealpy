#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:41, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec_basic import cec2014_nobias as cec
from pandas import DataFrame
from mealpy.evolutionary_based.DE import BaseDE
from os import getcwd, path, makedirs

model_name = "DE"
num_runs = 5
PATH_RESULTS = "history/results/"
check_dir1 = getcwd() + "/" + PATH_RESULTS
if not path.exists(check_dir1):
    makedirs(check_dir1)

## Setting parameters
verbose = True
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
for id_paras, func_name in enumerate(func_names):
    md = BaseDE(getattr(cec, func_name), lb1, ub1, verbose, epoch, pop_size, wf, cr)
    _, best_fit, list_loss = md.train()

    error_full[func_names[id_paras]] = list_loss
    error_columns.append(func_names[id_paras])

    best_fit_full[func_names[id_paras]] = [best_fit]
    best_fit_columns.append(func_names[id_paras])

df_err = DataFrame(error_full, columns=error_columns)
df_err.to_csv(PATH_RESULTS + str(len(lb1)) + "D_" + model_name + "_error.csv", header=True, index=False)

df_fit = DataFrame(best_fit_full, columns=best_fit_columns)
df_fit.to_csv(PATH_RESULTS + str(len(lb1)) + "D_" + model_name + "_best_fit.csv", header=True, index=False)

