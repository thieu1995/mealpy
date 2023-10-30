#!/usr/bin/env python
# Created by "Thieu" at 18:37, 27/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F52017
from mealpy import FloatVar, BBO

## Define your own problems
f1 = F52017(30, f_bias=0)

p1 = {
    "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
    "obj_func": f1.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": "console",
}

optimizer = BBO.OriginalBBO(epoch=100, pop_size=30)
optimizer.solve(p1)
