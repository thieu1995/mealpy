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
    "save_population": True
}

optimizer = BBO.OriginalBBO(epoch=100, pop_size=30)
optimizer.solve(p1, seed=10)        # Set seed for each solved problem

optimizer.history.save_diversity_chart()
optimizer.history.save_runtime_chart()
optimizer.history.save_trajectory_chart()
optimizer.history.save_exploration_exploitation_chart()
optimizer.history.save_global_best_fitness_chart()
optimizer.history.save_local_best_fitness_chart()
optimizer.history.save_global_objectives_chart()
optimizer.history.save_local_objectives_chart()

