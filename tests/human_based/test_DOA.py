#!/usr/bin/env python
# Created by "Madson" 21:23, 9/30/2026   ----------%
#       Email: madsonddias@gmail.com                %
#       Github: https://github.com/omadson          %
# --------------------------------------------------%

from mealpy import FloatVar, DOA, Optimizer
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


def test_DOA_results(problem):
    # pop_size is larger than the number of dimensions, and the
    # last 10% of epochs use the exploitation phase
    models = [
        DOA.OriginalDOA(epoch=10, pop_size=50),
    ]

    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
        assert np.all(g_best.solution >= model.problem.lb)
        assert np.all(g_best.solution <= model.problem.ub)
