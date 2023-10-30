#!/usr/bin/env python
# Created by "Thieu" at 16:05, 11/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F52017
from mealpy import FloatVar, BBO, Tuner


f1 = F52017(30, f_bias=0)

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": "console",
}

paras_bbo_grid = {
    "epoch": [10],
    "pop_size": [10],
    "n_elites": [2, 3, 4, 5],
    "p_m": [0.01, 0.02, 0.05]
}

term = {
    "max_epoch": 200,
    "max_time": 20,
    "max_fe": 10000
}

if __name__ == "__main__":
    model = BBO.OriginalBBO()
    tuner = Tuner(model, paras_bbo_grid)
    tuner.execute(problem=p1, termination=term, n_trials=5, n_jobs=4, verbose=True)

    print(tuner.best_row)
    print(tuner.best_score)
    print(tuner.best_params)
    print(type(tuner.best_params))

    print(tuner.best_algorithm)
    tuner.export_results()
    tuner.export_figures()

    g_best = tuner.resolve(mode="thread", n_workers=4, termination=term)
    print(g_best.solution, g_best.target.fitness)
    print(tuner.algorithm.problem.get_name())
    print(tuner.best_algorithm.get_name())
