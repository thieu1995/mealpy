#!/usr/bin/env python
# Created by "Thieu" at 13:00, 05/01/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from mealpy import FloatVar, SBOA, Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -10, -10, -10, -10], ub=[10, 10, 10, 10, 10]),
        "minmax": "min",
    }
    return problem


def test_OriginalSBOA_results(problem):
    epoch = 10
    pop_size = 50
    model = SBOA.OriginalSBOA(epoch, pop_size)
    g_best = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == len(model.problem.lb)


@pytest.mark.parametrize("problem, epoch, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -10, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, float("inf"), 0),
                         ])
def test_epoch_SBOA(problem, epoch, system_code):
    pop_size = 50
    with pytest.raises(ValueError) as e:
        SBOA.OriginalSBOA(epoch, pop_size)
    assert e.type == ValueError


@pytest.mark.parametrize("problem, pop_size, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -10, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, float("inf"), 0),
                         ])
def test_pop_size_SBOA(problem, pop_size, system_code):
    epoch = 10
    with pytest.raises(ValueError) as e:
        SBOA.OriginalSBOA(epoch, pop_size)
    assert e.type == ValueError
