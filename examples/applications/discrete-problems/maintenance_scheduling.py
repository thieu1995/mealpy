#!/usr/bin/env python
# Created by "Thieu" at 18:18, 05/11/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# In maintenance scheduling, the goal is to optimize the schedule for performing maintenance tasks on
# various assets or equipment. The objective is to minimize downtime and maximize the utilization of
# assets while considering various constraints such as resource availability, task dependencies,
# and time constraints.

# Each element in the solution represents whether a task is assigned to an asset (1) or not (0).
# The schedule specifies when each task should start and which asset it is assigned to,
# aiming to minimize the total downtime.
#
# By using the Mealpy, you can find an efficient maintenance schedule that minimizes downtime,
# maximizes asset utilization, and satisfies various constraints,
# ultimately optimizing the maintenance operations for improved reliability and productivity.


import numpy as np
from mealpy import BinaryVar, WOA, Problem


num_tasks = 10
num_assets = 5
task_durations = np.random.randint(1, 10, size=(num_tasks, num_assets))

data = {
    "num_tasks": num_tasks,
    "num_assets": num_assets,
    "task_durations": task_durations,
    "unassigned_penalty": -100         # Define a penalty value for no task is assigned to asset
}


class MaintenanceSchedulingProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["task_var"]
        downtime = -np.sum(x.reshape((self.data["num_tasks"], self.data["num_assets"])) * self.data["task_durations"])
        if np.sum(x) == 0:
            downtime += self.data["unassigned_penalty"]
        return downtime


bounds = BinaryVar(n_vars=num_tasks * num_assets, name="task_var")
problem = MaintenanceSchedulingProblem(bounds=bounds, minmax="max", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution).reshape((num_tasks, num_assets))}")      # Decoded (Real) solution
