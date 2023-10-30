#!/usr/bin/env python
# Created by "Thieu" at 11:41, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, BeesA, Optimizer
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
        "log_to": None
    }
    return problem


def test_BeesA_results(problem):
    models = [
        BeesA.ProbBeesA(epoch=10, pop_size=10, recruited_bee_ratio=0.1, dance_radius=0.1, dance_reduction=0.99),
        BeesA.OriginalBeesA(epoch=10, pop_size=10, selected_site_ratio=0.5, elite_site_ratio=0.4, selected_site_bee_ratio=0.1,
                            elite_site_bee_ratio=2.0, dance_radius=0.1, dance_reduction=0.99, ),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
