#!/usr/bin/env python
# Created by "Thieu" at 11:37, 18/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, QSA, Optimizer
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
        "log_to": None,
    }
    return problem


def test_QSA_results(problem):
    models = [
        QSA.DevQSA(epoch=10, pop_size=50),
        QSA.OriginalQSA(epoch=10, pop_size=50),
        QSA.OppoQSA(epoch=10, pop_size=50),
        QSA.LevyQSA(epoch=10, pop_size=50),
        QSA.ImprovedQSA(epoch=10, pop_size=50)
    ]

    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
