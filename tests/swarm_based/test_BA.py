#!/usr/bin/env python
# Created by "Thieu" at 01:14, 20/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest

from mealpy import FloatVar, BA, Optimizer


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


def test_BA_results(problem):
    models = [
        BA.OriginalBA(epoch=10, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0),
        BA.AdaptiveBA(epoch=10, pop_size=50, loudness_min=1.0, loudness_max=2.0, pr_min=0.15, pr_max=0.85, pf_min=-2.5, pf_max=10.),
        BA.DevBA(epoch=10, pop_size=50, pulse_rate=0.95, pf_min=0., pf_max=10.)
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
