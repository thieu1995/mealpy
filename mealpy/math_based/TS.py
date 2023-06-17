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

    Notes
    ~~~~~
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
    >>> from mealpy.math_based.TS import OriginalTS
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 2
    >>> tabu_size = 5
    >>> neighbour_size = 20
    >>> perturbation_scale = 0.05
    >>> model = OriginalTS(epoch, pop_size, tabu_size, neighbour_size, perturbation_scale)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Hajji, O., Brisset, S., & Brochet, P. (2004). A new tabu search method for optimization
    with continuous parameters. IEEE Transactions on Magnetics, 40(2), 1184-1187.
    """

    def __init__(self, epoch=10000, pop_size=2, tabu_size=5, neighbour_size=10, perturbation_scale=0.05, **kwargs):
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
        self.neighbour_size = self.validator.check_int("neighbour_size", neighbour_size, [2, 10000])
        self.tabu_size = self.validator.check_int("tabu_size", tabu_size, [2, 10000])
        self.perturbation_scale = self.validator.check_float("perturbation_scale", perturbation_scale, (0, 100))
        self.set_parameters(["epoch", "tabu_size", "neighbour_size", "perturbation_scale"])
        self.sort_flag = False

    def before_main_loop(self):
        self.x = self.g_best[self.ID_POS].copy()
        self.tabu_list = []

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Generate candidate solutions by perturbing the current solution
        candidates = np.random.normal(loc=self.x, scale=self.perturbation_scale,
                                      size=(self.neighbour_size, self.problem.n_dims))
        # Evaluate candidate solutions and select best move
        list_candidates = []
        for candidate in candidates:
            pos_new = self.amend_position(candidate, self.problem.lb, self.problem.ub)
            if np.allclose(pos_new, self.x):
                continue
            if tuple(pos_new) in self.tabu_list:
                continue
            list_candidates.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                list_candidates[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        list_candidates = self.update_target_wrapper_population(list_candidates)
        self.pop, best_candidate = self.get_global_best_solution(list_candidates)
        self.x = best_candidate[self.ID_POS]

        # Update tabu list
        self.tabu_list.append(tuple(self.x))
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
