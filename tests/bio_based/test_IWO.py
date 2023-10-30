#!/usr/bin/env python
# Created by "Thieu" at 19:12, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, IWO, Optimizer
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


def test_OriginalIWO_results(problem):
    epoch = 10
    pop_size = 50
    seed_min = 2
    seed_max = 10
    exponent = 2
    sigma_start = 1.0
    sigma_end = 0.01

    model = IWO.OriginalIWO(epoch, pop_size, seed_min, seed_max, exponent, sigma_start, sigma_end)
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
def test_epoch_IWO(problem, epoch, system_code):
    pop_size = 50
    algorithms = [IWO.OriginalIWO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
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
def test_pop_size_IWO(problem, pop_size, system_code):
    epoch = 10
    algorithms = [IWO.OriginalIWO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, exponent, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -1.0, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 1, 0),
                             (problem, 50, 0),
                             (problem, 100, 0),
                             (problem, 1.6, 0),
                         ])
def test_exponent_IWO(problem, exponent, system_code):
    algorithms = [IWO.OriginalIWO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, exponent=exponent)
        assert e.type == ValueError
