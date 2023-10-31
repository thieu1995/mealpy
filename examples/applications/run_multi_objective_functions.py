#!/usr/bin/env python
# Created by "Thieu" at 17:40, 06/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar, SMA
import numpy as np


## Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
def objective_function(solution):

    def booth(x, y):
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2

    def bukin(x, y):
        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

    def matyas(x, y):
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    return [booth(solution[0], solution[1]), bukin(solution[0], solution[1]), matyas(solution[0], solution[1])]


problem = {
    "obj_func": objective_function,
    "bounds": FloatVar(lb=(-10, -10), ub=(10, 10)),
    "minmax": "min",
    "obj_weights": [0.4, 0.1, 0.5]               # Define it or default value will be [1, 1, 1]
}

## Run the algorithm
optimizer = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
optimizer.solve(problem)
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")

## You can access all of available figures via object "history" like this:
optimizer.history.save_global_objectives_chart(filename="hello/goc")
optimizer.history.save_local_objectives_chart(filename="hello/loc")
optimizer.history.save_global_best_fitness_chart(filename="hello/gbfc")
optimizer.history.save_local_best_fitness_chart(filename="hello/lbfc")
optimizer.history.save_runtime_chart(filename="hello/rtc")
optimizer.history.save_exploration_exploitation_chart(filename="hello/eec")
optimizer.history.save_diversity_chart(filename="hello/dc")
optimizer.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[2], filename="hello/tc")


from mealpy import Problem
import numpy as np

# Our custom problem class
class Squared(Problem):
    def __init__(self, bounds=None, minmax="min", name="Squared", data=None, **kwargs):
        self.name = name
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, solution):
        return np.sum(solution ** 2)
