#!/usr/bin/env python
# Created by "Thieu" at 10:50, 17/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pandas import DataFrame
from mealpy.swarm_based import WOA
from timeseries_util import decode_solution, generate_loss_value, generate_data
from sklearn.preprocessing import LabelEncoder
from os import getcwd, path, makedirs
import time
import numpy as np
np.random.seed(12345)


def fitness_function(solution, data):
    structure = decode_solution(solution, data)
    fitness = generate_loss_value(structure, data)
    return fitness


if __name__ == "__main__":

    # LABEL ENCODER
    OPT_ENCODER = LabelEncoder()
    OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  # domain range ==> 7 values

    WOI_ENCODER = LabelEncoder()
    WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

    ACT_ENCODER = LabelEncoder()
    ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])

    DATA = generate_data()
    DATA["OPT_ENCODER"] = OPT_ENCODER
    DATA["WOI_ENCODER"] = WOI_ENCODER
    DATA["ACT_ENCODER"] = ACT_ENCODER

    model_name = "WOA"
    N_TRIALS = 1
    LB = [1, 5, 0, 0.01, 0, 0, 5]
    UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 50]
    epoch = 50
    pop_size = 20
    mode_names = ["single", "swarm", "thread", "process"]

    PATH_ERROR = "history/error/" + model_name + "/"
    PATH_BEST_FIT = "history/best_fit/"
    check_dir1 = f"{getcwd()}/{PATH_ERROR}"
    check_dir2 = f"{getcwd()}/{PATH_BEST_FIT}"
    if not path.exists(check_dir1): makedirs(check_dir1)
    if not path.exists(check_dir2): makedirs(check_dir2)

    ## Run model
    best_fit_full = {}
    best_fit_columns = []
    list_total_time = []

    for mode_name in mode_names:
        error_full = {}
        error_columns = []
        best_fit_list = []

        for id_trial in range(1, N_TRIALS + 1):
            time_start = time.perf_counter()
            problem = {
                "fit_func": fitness_function,
                "lb": LB,
                "ub": UB,
                "minmax": "min",
                "log_to": None,
                "save_population": False,
                "data": DATA,
            }
            model = WOA.BaseWOA(problem, epoch, pop_size)
            _, best_fitness = model.solve(mode=mode_name)
            time_end = time.perf_counter() - time_start

            temp = f"trial_{id_trial}"
            error_full[temp] = model.history.list_global_best_fit
            error_columns.append(temp)
            best_fit_list.append(best_fitness)

            list_total_time.append([mode_name, id_trial, time_end])

        df = DataFrame(error_full, columns=error_columns)

        df.to_csv(f"{PATH_ERROR}{len(LB)}D_{model_name}_{mode_name}_mlp_paras_tuning_error.csv", header=True, index=False)
        best_fit_full[mode_name] = best_fit_list
        best_fit_columns.append(mode_name)

    df = DataFrame(best_fit_full, columns=best_fit_columns)
    df.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_mlp_paras_tuning_best_fit.csv", header=True, index=False)

    df_time = DataFrame(np.array(list_total_time), columns=["mode", "trial", "total_time"])
    df_time.to_csv(f"{PATH_BEST_FIT}/{len(LB)}D_{model_name}_mlp_paras_tuning_total_time.csv", header=True, index=False)
