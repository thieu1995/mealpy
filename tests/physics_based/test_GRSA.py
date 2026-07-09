from mealpy import FloatVar, Optimizer
from mealpy.physics_based import GRSA
import numpy as np
import pytest


@pytest.fixture(scope="module")  # scope: Call only 1 time at the beginning
def problem():
    def objective_function(solution):
        return np.sum(solution ** 2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(
            lb=[-10, -15, -4, -2, -8],
            ub=[10, 15, 12, 8, 20],
        ),
        "minmax": "min",
    }
    return problem


def test_GRSA_results(problem):
    models = [
        GRSA.OriginalGRSA(
            epoch=100,
            pop_size=50,
            w_max=0.9,
            w_min=0.1,
            k_g=0.5,
            mutation_rate=0.1,
        ),
    ]
    for model in models:
        g_best = model.solve(problem)
        assert isinstance(model, Optimizer)
        assert isinstance(g_best.solution, np.ndarray)
        assert len(g_best.solution) == len(model.problem.lb)
