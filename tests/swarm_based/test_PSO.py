#!/usr/bin/env python
# Created by "Thieu" at 10:54, 21/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, PSO, Optimizer
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


def test_PSO_results(problem):
    models = [
        PSO.OriginalPSO(epoch=100, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
        PSO.C_PSO(epoch=10, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9),
        PSO.CL_PSO(epoch=10, pop_size=50, c_local=1.2, w_min=0.4, w_max=0.9, max_flag=7),
        PSO.P_PSO(epoch=10, pop_size=50),
        PSO.HPSO_TVAC(epoch=10, pop_size=50, ci=0.5, cf=0.2)
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
