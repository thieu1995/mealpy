#!/usr/bin/env python
# Created by "Thieu" at 21:23, 07/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F52017, F102017, F292017
from mealpy.bio_based import BBO
from mealpy.evolutionary_based import DE
from mealpy.utils.app import Multitask


#### Using multiple algorithm to solve multiple problems with multiple trials

f1 = F52017(30, f_bias=0)
f2 = F102017(30, f_bias=0)
f3 = F292017(30, f_bias=0)

p1 = {
    "lb": f1.lb.tolist(),
    "ub": f1.ub.tolist(),
    "minmax": "min",
    "fit_func": f1.evaluate,
    "name": "F5",
    "log_to": None,
}

p2 = {
    "lb": f2.lb.tolist(),
    "ub": f2.ub.tolist(),
    "minmax": "min",
    "fit_func": f2.evaluate,
    "name": "F10",
    "log_to": None,
}

p3 = {
    "lb": f3.lb.tolist(),
    "ub": f3.ub.tolist(),
    "minmax": "min",
    "fit_func": f3.evaluate,
    "name": "F29",
    "log_to": None,
}


model1 = BBO.BaseBBO(epoch=10, pop_size=50)
model2 = BBO.OriginalBBO(epoch=10, pop_size=50)
model3 = DE.BaseDE(epoch=10, pop_size=50)


multitask = Multitask(algorithms=(model1, model2, model3), problems=(p1, p2, p3), n_trials=3)
multitask.execute(save_path="history", save_as="dataframe", save_convergence=True, verbose=True)
