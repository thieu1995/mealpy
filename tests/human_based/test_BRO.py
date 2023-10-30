#!/usr/bin/env python
# Created by "Thieu" at 08:50, 18/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, BRO, Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
    }
    return problem


def test_BRO_results(problem):
    models = [
        BRO.OriginalBRO(epoch=100, pop_size=50, threshold=1),
        BRO.DevBRO(epoch=10, pop_size=50, threshold=1),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
