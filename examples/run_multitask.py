#!/usr/bin/env python
# Created by "Thieu" at 21:23, 07/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

#### Using multiple algorithm to solve multiple problems with multiple trials


## Import libraries

from opfunu.cec_based.cec2017 import F52017, F102017, F292017
from mealpy.bio_based import BBO
from mealpy.evolutionary_based import DE
from mealpy.multitask import Multitask


## Define your own problems

f1 = F52017(30, f_bias=0)
f2 = F102017(30, f_bias=0)
f3 = F292017(30, f_bias=0)

p1 = {
    "lb": f1.lb,
    "ub": f1.ub,
    "minmax": "min",
    "fit_func": f1.evaluate,
    "name": "F5",
    "log_to": None,
}

p2 = {
    "lb": f2.lb,
    "ub": f2.ub,
    "minmax": "min",
    "fit_func": f2.evaluate,
    "name": "F10",
    "log_to": None,
}

p3 = {
    "lb": f3.lb,
    "ub": f3.ub,
    "minmax": "min",
    "fit_func": f3.evaluate,
    "name": "F29",
    "log_to": None,
}

## Define models

model1 = BBO.BaseBBO(epoch=10000, pop_size=50)
model2 = BBO.OriginalBBO(epoch=10000, pop_size=50)
model3 = DE.BaseDE(epoch=10000, pop_size=50)
model4 = DE.SAP_DE(epoch=10000, pop_size=50)

## Define termination if needed
term = {
    "max_fe": 30000
}

## Define and run Multitask

if __name__ == "__main__":
    multitask = Multitask(algorithms=(model1, model2, model3, model4), problems=(p1, p2, p3), terminations=(term, ), modes=("thread", ))
    # default modes = "single", default termination = epoch (as defined in problem dictionary)
    multitask.execute(n_trials=5, n_jobs=5, save_path="history", save_as="csv", save_convergence=True, verbose=False)
    # multitask.execute(n_trials=5, save_path="history", save_as="csv", save_convergence=True, verbose=False)

