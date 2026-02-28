#!/usr/bin/env python
# Created by "Thieu" at 11:10, 17/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import pytest

from mealpy import FloatVar, BWO, Optimizer


@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    return {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
    }


def test_BWO_results(problem):
    models = [
        BWO.OriginalBWO(epoch=10, pop_size=50, pp=0.6, cr=0.44, pm=0.4),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
