#!/usr/bin/env python
# Created by "Thieu" at 00:16, 15/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, BBO, Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)
    prob = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10, -15, -4, -2, -8], ub=[10, 15, 12, 8, 20]),
        "minmax": "min",
        "log_to": None,
    }
    return prob


def test_OriginalBBO_results(problem):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    n_elites = 2
    model = BBO.OriginalBBO(epoch, pop_size, p_m, n_elites)
    g_best = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
    assert len(g_best.solution) == len(model.problem.lb)


def test_DevBBO_results(problem):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    n_elites = 3
    model = BBO.DevBBO(epoch, pop_size, p_m, n_elites)
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
def test_epoch_BBO(problem, epoch, system_code):
    pop_size = 50
    p_m = 0.01
    n_elites = 2
    algorithms = [BBO.OriginalBBO, BBO.DevBBO]
    for algorithm in algorithms:
        with pytest.raises(Exception) as e:
            algorithm(epoch, pop_size, p_m, n_elites)
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
def test_pop_size_BBO(problem, pop_size, system_code):
    epoch = 10
    p_m = 0.01
    n_elites = 2
    algorithms = [BBO.OriginalBBO, BBO.DevBBO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size, p_m, n_elites)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, p_m, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, -1.0, 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 0, 0),
                             (problem, 1, 0),
                             (problem, 1.1, 0),
                             (problem, -0.01, 0),
                         ])
def test_p_m_BBO(problem, p_m, system_code):
    epoch = 10
    pop_size = 50
    n_elites = 2
    algorithms = [BBO.OriginalBBO, BBO.DevBBO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size, p_m, n_elites)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, n_elites, system_code",
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
def test_n_elites_BBO(problem, n_elites, system_code):
    epoch = 10
    pop_size = 50
    p_m = 0.01
    algorithms = [BBO.OriginalBBO, BBO.DevBBO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size, p_m, n_elites)
        assert e.type == ValueError
