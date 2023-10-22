#!/usr/bin/env python
# Created by "Thieu" at 06:25, 17/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTS(Optimizer):
    """
    The original version of: Tabu Search (TS)

    Notes:
        + The pop_size is not an official parameter in this algorithm. However, we need it here to adapt to Mealpy library.
        + You should set pop_size = 2 to reduce the initial computation for the initial population of this algorithm.
        + The perturbation_scale is important parameter that effect the most to this algorithm.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + tabu_size (int): [5, 10] Maximum size of the tabu list.
        + neighbour_size (int): [5, 100] Size of the neighborhood for generating candidate solutions, Default: 10
        + perturbation_scale (float): [0.01 - 1] Scale of the perturbations for generating candidate solutions. default = 0.05

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TS
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = TS.OriginalTS(epoch=1000, pop_size=50, tabu_size = 5, neighbour_size = 20, perturbation_scale = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hajji, O., Brisset, S., & Brochet, P. (2004). A new tabu search method for optimization
    with continuous parameters. IEEE Transactions on Magnetics, 40(2), 1184-1187.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 2, tabu_size: int = 5,
                 neighbour_size: int = 10, perturbation_scale: float = 0.05, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size. **This is not an official parameter**. So default = 2.
            tabu_size (int): Maximum size of the tabu list.
            neighbour_size (int): Size of the neighborhood for generating candidate solutions, Default: 10
            perturbation_scale (float): Scale of the perturbations for generating candidate solutions. default = 0.05
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.tabu_size = self.validator.check_int("tabu_size", tabu_size, [2, 10000])
        self.neighbour_size = self.validator.check_int("neighbour_size", neighbour_size, [2, 10000])
        self.perturbation_scale = self.validator.check_float("perturbation_scale", perturbation_scale, (0, 100))
        self.set_parameters(["epoch", "pop_size", "tabu_size", "neighbour_size", "perturbation_scale"])
        self.sort_flag = False

    def before_main_loop(self):
        self.x = self.g_best.solution.copy()
        self.tabu_list = []

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Generate candidate solutions by perturbing the current solution
        candidates = self.generator.normal(loc=self.x, scale=self.perturbation_scale, size=(self.neighbour_size, self.problem.n_dims))
        # Evaluate candidate solutions and select best move
        list_candidates = []
        for candidate in candidates:
            pos_new = self.correct_solution(candidate)
            if np.allclose(pos_new, self.x):
                continue
            if tuple(pos_new) in self.tabu_list:
                continue
            agent = self.generate_empty_agent(pos_new)
            list_candidates.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                list_candidates[-1].target = self.get_target(pos_new)
        list_candidates = self.update_target_for_population(list_candidates)
        best_candidate = self.get_best_agent(list_candidates, self.problem.minmax)
        self.x = best_candidate.solution
        # Update tabu list
        self.tabu_list.append(tuple(self.x))
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
