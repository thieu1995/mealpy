#!/usr/bin/env python
# Created by "Furkan Buyukyozgat" at 15:25, 05/01/2026-------%
#       Email: furkanbuyuky@gmail.com                        %
#       Github: https://github.com/furkanbuyuky              %
# -----------------------------------------------------------%

import numpy as np
import pytest

from mealpy import FloatVar, OSA, Optimizer


@pytest.fixture(scope="module")
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


def test_OSA_results(problem):
    model = OSA.OriginalOSA(epoch=10, pop_size=50, beta_max=1.9, alpha_max=0.5)
    g_best = model.solve(problem)

    assert isinstance(model, Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == len(model.problem.lb)
