#!/usr/bin/env python
# Created by "Thieu" at 00:55, 17/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.bio_based import WHO
from mealpy.optimizer import Optimizer
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def fitness_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "fit_func": fitness_function,
        "lb": [-10, -10, -10, -10, -10],
        "ub": [10, 10, 10, 10, 10],
        "minmax": "min",
    }
    return problem


def test_BaseWHO_results(problem):
    epoch = 10
    pop_size = 50
    n_s = 3
    n_e = 3
    eta = 0.15
    p_hi = 0.9
    model = WHO.OriginalWHO(epoch, pop_size, n_s, n_e, eta, p_hi)
    best_position, best_fitness = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


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
def test_epoch_WHO(problem, epoch, system_code):
    pop_size = 50
    algorithms = [WHO.OriginalWHO]
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
def test_pop_size_WHO(problem, pop_size, system_code):
    epoch = 10
    algorithms = [WHO.OriginalWHO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(epoch, pop_size)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, p_hi, system_code",
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
def test_p_hi_WHO(problem, p_hi, system_code):
    algorithms = [WHO.OriginalWHO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(50, p_hi=p_hi)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, eta, system_code",
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
def test_eta_WHO(problem, eta, system_code):
    algorithms = [WHO.OriginalWHO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(50, eta=eta)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, n_explore_step, system_code",
                         [
                             (problem, None, 0),
                             (problem, "hello", 0),
                             (problem, [10], 0),
                             (problem, (0, 9), 0),
                             (problem, 1, 0),
                             (problem, 50, 0),
                             (problem, 100, 0),
                             (problem, 1.6, 0),
                         ])
def test_n_s_WHO(problem, n_explore_step, system_code):
    algorithms = [WHO.OriginalWHO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(50, n_explore_step=n_explore_step)
        assert e.type == ValueError


@pytest.mark.parametrize("problem, n_exploit_step, system_code",
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
def test_n_e_WHO(problem, n_exploit_step, system_code):
    algorithms = [WHO.OriginalWHO]
    for algorithm in algorithms:
        with pytest.raises(ValueError) as e:
            algorithm(50, n_exploit_step=n_exploit_step)
        assert e.type == ValueError
