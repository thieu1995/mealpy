#!/usr/bin/env python
# Created by "Thieu" at 11:30, 05/01/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar
from mealpy import ICSA
from mealpy import Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem_dict = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None
    }
    return problem_dict


def test_OriginalICSA_results(problem):
    models = [
        ICSA.OriginalICSA(epoch=10, pop_size=50, beta=1.5, r_chaos=0.3, k_spiral=5)
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)