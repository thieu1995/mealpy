#!/usr/bin/env python
# Created by "Thieu" at 19:45, 19/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar, KOA, Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    return {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None,
    }


def test_KOA_results(problem):
    model = KOA.OriginalKOA(epoch=25, pop_size=15, T=3, gamma=15, mu0=0.1)
    g_best = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == len(model.problem.lb)
