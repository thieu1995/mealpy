import numpy as np

from mealpy import FloatVar
from mealpy.swarm_based import RSA


def test_RSA_should_work():
    def obj(sol):
        return np.sum(sol**2)

    problem = {
        "bounds": FloatVar(lb=(-5.,) * 10, ub=(5.,) * 10, name="x"),
        "minmax": "min",
        "obj_func": obj,
    }

    model = RSA.OriginalRSA(epoch=20, pop_size=20, alpha=0.1, beta=0.1)
    g_best = model.solve(problem)

    assert g_best is not None
    assert g_best.solution is not None
    assert np.isfinite(g_best.target.fitness)
