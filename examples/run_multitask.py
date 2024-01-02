#!/usr/bin/env python
# Created by "Thieu" at 21:23, 07/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

#### Using multiple algorithm to solve multiple problems with multiple trials

## Import libraries
from opfunu.cec_based.cec2017 import F52017, F102017, F292017
from mealpy import FloatVar
from mealpy import BBO, DE
from mealpy import Multitask

## Define your own problems
f1 = F52017(30, f_bias=0)
f2 = F102017(30, f_bias=0)
f3 = F292017(30, f_bias=0)

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": "console",
}

p2 = {
    "bounds": FloatVar(lb=f2.lb, ub=f2.ub),
    "obj_func": f2.evaluate,
    "minmax": "min",
    "name": "F10",
    "log_to": "console",
}

p3 = {
    "bounds": FloatVar(lb=f3.lb, ub=f3.ub),
    "obj_func": f3.evaluate,
    "minmax": "min",
    "name": "F29",
    "log_to": "console",
}

## Define models
model1 = BBO.DevBBO(epoch=10000, pop_size=50)
model2 = BBO.OriginalBBO(epoch=10000, pop_size=50)
model3 = DE.OriginalDE(epoch=10000, pop_size=50)
model4 = DE.SAP_DE(epoch=10000, pop_size=50)

## Define termination if needed
term = {
    "max_fe": 3000
}

## Define and run Multitask
if __name__ == "__main__":
    multitask = Multitask(algorithms=(model1, model2, model3, model4), problems=(p1, p2, p3), terminations=(term, ), modes=("thread", ), n_workers=4)
    # default modes = "single", default termination = epoch (as defined in problem dictionary)
    multitask.execute(n_trials=5, n_jobs=None, save_path="history7", save_as="csv", save_convergence=True, verbose=False)
    # multitask.execute(n_trials=5, save_path="history", save_as="csv", save_convergence=True, verbose=False)
