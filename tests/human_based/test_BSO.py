#!/usr/bin/env python
# Created by "Thieu" at 08:59, 18/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, BSO, Optimizer
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


def test_BSO_results(problem):
    models = [
        BSO.OriginalBSO(epoch=100, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5),
        BSO.ImprovedBSO(epoch=10, pop_size=50, m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, slope=30),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
