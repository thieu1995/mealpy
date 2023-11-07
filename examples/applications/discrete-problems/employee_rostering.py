#!/usr/bin/env python
# Created by "Thieu" at 17:48, 05/11/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

## The goal is to create an optimal schedule that assigns employees to shifts while satisfying various constraints and objectives.
# Note that this implementation assumes that the shift requirements and costs are provided as numpy arrays. The
#   shift_requirements array has dimensions (num_employees, num_shifts), and shift_costs is a 1D array of length num_shifts.
# Please keep in mind that this is a simplified implementation, and you may need to modify it according to the
#   specific requirements and constraints of your employee rostering problem.
# Additionally, you might want to introduce additional mechanisms or constraints such as fairness, employee
#   preferences, or shift dependencies to enhance the model's effectiveness in real-world scenarios.


# For example, if you have 5 employees and 3 shifts, a chromosome could be represented as [2, 1, 0, 2, 0],
# where employee 0 is assigned to shift 2, employee 1 is assigned to shift 1, employee 2 is assigned to shift 0, and so on.


import numpy as np
from mealpy import IntegerVar, WOA, Problem


shift_requirements = np.array([[2, 1, 3], [4, 2, 1], [3, 3, 2]])
shift_costs = np.array([10, 8, 12])

num_employees = shift_requirements.shape[0]
num_shifts = shift_requirements.shape[1]

data = {
    "shift_requirements": shift_requirements,
    "shift_costs": shift_costs,
    "num_employees": num_employees,
    "num_shifts": num_shifts
}

class EmployeeRosteringProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["shift_var"]

        shifts_covered = np.zeros(self.data["num_shifts"])
        total_cost = 0
        for idx in range(self.data["num_employees"]):
            shift_idx = x[idx]
            shifts_covered[shift_idx] += 1
            total_cost += self.data["shift_costs"][shift_idx]
        coverage_diff = self.data["shift_requirements"] - shifts_covered
        coverage_penalty = np.sum(np.abs(coverage_diff))
        return total_cost + coverage_penalty


bounds = IntegerVar(lb=[0, ]*num_employees, ub=[num_shifts-1, ]*num_employees, name="shift_var")
problem = EmployeeRosteringProblem(bounds=bounds, minmax="min", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")      # Decoded (Real) solution
