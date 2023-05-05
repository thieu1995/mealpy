#!/usr/bin/env python
# Created by "Thieu" at 16:05, 11/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F52017
from mealpy.bio_based import BBO
from mealpy.tuner import Tuner

f1 = F52017(30, f_bias=0)

p1 = {
    "lb": f1.lb,
    "ub": f1.ub,
    "minmax": "min",
    "fit_func": f1.evaluate,
    "name": "F5",
    "log_to": None,
}

paras_bbo_grid = {
    "epoch": [100],
    "pop_size": [50],
    "elites": [2, 3, 4, 5],
    "p_m": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
}

term = {
    "max_epoch": 200,
    "max_time": 20,
    "max_fe": 10000
}

if __name__ == "__main__":
    model = BBO.BaseBBO()
    tuner = Tuner(model, paras_bbo_grid)
    tuner.execute(problem=p1, termination=term, n_trials=5, n_jobs=4, mode="single", n_workers=6, verbose=True)

    print(tuner.best_row)
    print(tuner.best_score)
    print(tuner.best_params)
    print(type(tuner.best_params))

    print(tuner.best_algorithm)
    tuner.export_results("history/tuning", save_as="csv")

    best_position, best_fitness = tuner.resolve(mode="thread", n_workers=4, termination=term)
    print(best_position, best_fitness)
    print(tuner.problem.get_name())
    print(tuner.best_algorithm.get_name())
