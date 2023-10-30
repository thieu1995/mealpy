#!/usr/bin/env python
# Created by "Thieu" at 18:13, 16/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import FloatVar, EOA, Optimizer
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


def test_BaseEOA_results(problem):
    epoch = 10
    pop_size = 50
    p_c = 0.9
    p_m = 0.01
    n_best = 2
    alpha = 0.98
    beta = 0.9
    gama = 0.9
    model = EOA.OriginalEOA(epoch, pop_size, p_c, p_m, n_best, alpha, beta, gama)
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
def test_epoch_EOA(problem, epoch, system_code):
    pop_size = 50
    algorithms = [EOA.OriginalEOA]
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
def test_pop_size_EOA(problem, pop_size, system_code):
    epoch = 10
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, p_c, system_code",
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
def test_p_c_EOA(problem, p_c, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, p_c=p_c)
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
def test_p_m_EOA(problem, p_m, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, p_m=p_m)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, n_best, system_code",
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
def test_n_best_EOA(problem, n_best, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, n_best=n_best)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, alpha, system_code",
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
def test_alpha_EOA(problem, alpha, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, alpha=alpha)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, beta, system_code",
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
def test_beta_EOA(problem, beta, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, beta=beta)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, gama, system_code",
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
def test_gama_EOA(problem, gama, system_code):
    algorithms = [EOA.OriginalEOA]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(10, 50, gama=gama)
        assert e.type == ValueError
