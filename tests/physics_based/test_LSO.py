#!/usr/bin/env python
# Created by "ayp" at 2024 ----------%
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar, LSO, Optimizer
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


def test_OriginalLSO_results(problem):
    models = [
        LSO.OriginalLSO(epoch=100, pop_size=50, Ps=0.05, Pe=0.6, Ph=0.4, B=0.05),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)


def test_DevLSO_results(problem):
    models = [
        LSO.DevLSO(epoch=100, pop_size=50, Ps=0.05, Pe=0.6, B=0.05),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)