#!/usr/bin/env python
# Created by "Thieu" at 09:34, 05/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy import PermutationVar, ACOR, Problem


class MyProblem(Problem):
    def __init__(self, bounds, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        order = self.decode_solution(x)["delta"]

        t = np.zeros((self.data["n_jobs"], self.data["n_machines"]))
        for job in range(self.data["n_jobs"]):
            for machine in range(self.data["n_machines"]):
                if machine==0 and job ==0:
                    t[job,machine] = self.data["p"][int(order[job]),machine]
                elif machine==0:
                    t[job,machine]=t[job-1, machine]+self.data["p"][int(order[job]),machine]
                elif job==0:
                    t[job,machine]=t[job, machine-1]+self.data["p"][int(order[job]),machine]
                else:
                    t[job,machine]=max(t[job-1, machine],t[job, machine-1])+self.data["p"][int(order[job]),machine]
        makespan=t[-1,-1]
        return makespan


n_jobs = 5
n_machines = 4
data = {
    "p": np.array([[4, 3, 6, 2], [1, 4, 3, 5], [2, 5, 2, 3], [5, 2, 4, 1], [3, 6, 1, 4]]),
    "order": list(range(0, n_jobs)),
    "n_jobs": n_jobs,
    "machines": list(range(0, n_machines)),
    "n_machines": n_machines,

}

problem = MyProblem(bounds=PermutationVar(valid_set=(0, 1, 2, 3, 4), name="delta"), name="Wow",
                    minmax="min", data=data, log_to="console")

model = ACOR.OriginalACOR(epoch=50, pop_size=20, sample_count = 25, intent_factor = 0.5, zeta = 1.0)
g_best = model.solve(problem)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
print(f"The real solution: {problem.decode_solution(g_best.solution)['delta']}")
print(problem.get_name())
print(model.problem.get_name())
