# !/usr/bin/env python
# Created by "Thieu" at 15:49, 10/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy.bio_based import SMA
import numpy as np


def fitness_function(solution):
    return np.sum(solution**2)


problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-100, ] * 30,
    "ub": [100, ] * 30,
    "minmax": "min",
}

## Run the algorithm
model1 = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
best_position, best_fitness = model1.solve(problem_dict1)
print(f"Best solution: {best_position}, Best fitness: {best_fitness}")