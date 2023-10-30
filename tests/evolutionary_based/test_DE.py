#!/usr/bin/env python
# Created by "Thieu" at 09:35, 17/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, DE, Optimizer
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


def test_DE_results(problem):
    models = [
        DE.OriginalDE(epoch=10, pop_size=50, wf=0.1, cr=0.9, strategy=5),
        DE.JADE(epoch=20, pop_size=50, miu_f=0.5, miu_cr=0.5, pt=0.1, ap=0.1),
        DE.SADE(epoch=20, pop_size=50),
        DE.SAP_DE(epoch=20, pop_size=50, branch="ABS"),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
