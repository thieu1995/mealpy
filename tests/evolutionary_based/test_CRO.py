#!/usr/bin/env python
# Created by "Thieu" at 08:33, 17/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.evolutionary_based import CRO
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
        "log_to": None,
    }
    return problem


def test_OriginalCRO_results(problem):
    epoch = 10
    pop_size = 50
    po = 0.4
    Fb = 0.9
    Fa = 0.1
    Fd = 0.1
    Pd = 0.1
    GCR = 0.1
    gamma_min = 0.02
    gamma_max = 0.2
    n_trials = 5
    model = CRO.OriginalCRO(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials)
    best_position, best_fitness = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


def test_OCRO_results(problem):
    epoch = 10
    pop_size = 50
    po = 0.4
    Fb = 0.9
    Fa = 0.1
    Fd = 0.1
    Pd = 0.1
    GCR = 0.1
    gamma_min = 0.02
    gamma_max = 0.2
    n_trials = 5
    model = CRO.OriginalCRO(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials)
    best_position, best_fitness = model.solve(problem)
    assert isinstance(model, Optimizer)
    assert isinstance(best_position, np.ndarray)
    assert len(best_position) == len(problem["lb"])


def test_params_CRO(problem):
    epoch = 10
    pop_size = 50
    po = 0.4
    Fb = 0.9
    Fa = 0.1
    Fd = 0.1
    Pd = 0.1
    GCR = 0.1
    gamma_min = 0.02
    gamma_max = 0.2
    n_trials = 5
    restart_count = 3
    model = CRO.OCRO(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials, restart_count)
    assert model.po == po
    assert model.Fb == Fb
    assert model.Fa == Fa
    assert model.Fd == Fd
    assert model.Pd == Pd
    assert model.GCR == GCR
    assert model.gamma_max == gamma_max
    assert model.gamma_min == gamma_min
    assert model.n_trials == n_trials
    assert model.restart_count == restart_count
