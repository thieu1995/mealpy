#!/usr/bin/env python
# Created by "Thieu" at 16:09, 24/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based.cec2017 import F52017
from mealpy import FloatVar, BBO


# Find the index of population that contains the global best solution.
def get_first_best_population(opt):
    for idx, pop_child in enumerate(opt.history.list_population):
        for agent in pop_child:
            if agent.target.fitness == opt.g_best.target.fitness:
                return idx, pop_child
    return None, None

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

idx, my_first_found_best_pop = get_first_best_population(optimizer)
print(f"The global best solution found at generation: {idx}")
print(my_first_found_best_pop)
