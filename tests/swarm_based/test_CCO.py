import numpy as np

from mealpy import FloatVar
from mealpy.swarm_based.CCO import OriginalCCO


def test_cco_runs_and_returns_best():
    def obj(solution):
        return np.sum(solution**2)

    problem = {
        "bounds": FloatVar(lb=(-5.0,) * 10, ub=(5.0,) * 10, name="x"),
        "minmax": "min",
        "obj_func": obj,
    }

    model = OriginalCCO(epoch=30, pop_size=20)
    best = model.solve(problem)

    assert best is not None
    assert hasattr(best, "solution")
    assert hasattr(best, "target")
    assert best.solution.shape == (10,)
    assert np.isfinite(best.target.fitness)
