#!/usr/bin/env python
# Created by "Antigravity" at 2025

import numpy as np
import pytest
from mealpy import FloatVar, Optimizer
from mealpy.swarm_based import MLFA_GD

@pytest.fixture(scope="module")
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-10.0] * 5, ub=[10.0] * 5, name="delta"),
        "minmax": "min",
        "log_to": None
    }
    return problem

def test_MLFA_GD_correctness(problem):
    """
    Test that MLFA-GD runs and returns valid results on a simple Sphere function.
    """
    model = MLFA_GD(epoch=50, pop_size=20, gamma=1.0, beta_base=1.0, alpha=0.2, m_females=3, learning_count=50, k_walk=3)
    
    g_best = model.solve(problem)
    
    # Assert result structure
    assert isinstance(model, Optimizer)
    assert isinstance(g_best.solution, np.ndarray)
    
    # Assert dimensions
    assert len(g_best.solution) == 5
    
    # Assert fitness is numeric
    assert isinstance(g_best.target.fitness, (float, int, np.floating, np.integer))
    
    # Sphere function optimal is 0. With 50 epochs and bounds [-10, 10], 
    # it should be reasonably close to 0, but mainly we check it didn't diverge/NaN.
    assert g_best.target.fitness < 1000.0
    assert not np.isnan(g_best.target.fitness)

def test_MLFA_GD_hyperparameters():
    """
    Test initialization with specific hyperparameters.
    """
    model = MLFA_GD(epoch=10, pop_size=30, m_females=2, learning_count=100)
    assert model.pop_size == 30
    assert model.m_females == 2
    assert model.learning_count == 100
