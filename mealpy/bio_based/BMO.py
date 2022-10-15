#!/usr/bin/env python
# Created by "Thieu" at 11:10, 15/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBMO(Optimizer):
    """
    The original version: Barnacles Mating Optimizer (BMO)

    Links:
        1. https://ieeexplore.ieee.org/document/8441097

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pl: [1, pop_size - 1], barnacle’s threshold

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.BMO import OriginalBMO
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
    >>> pop_size = 50
    >>> pl = 4
    >>> model = OriginalBMO(epoch, pop_size, pl)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    def __init__(self, epoch=10000, pop_size=100, pl=5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pl = self.validator.check_int("pl", pl, [1, self.pop_size-1])
        self.set_parameters(["epoch", "pop_size", "pl"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        k1 = np.random.permutation(self.pop_size)
        k2 = np.random.permutation(self.pop_size)
        temp = np.abs(k1 - k2)
        pop_new = []
        for idx in range(0, self.pop_size):
            if temp[idx] <= self.pl:
                p = np.random.uniform(0, 1)
                pos_new = p * self.pop[k1[idx]][self.ID_POS] + (1 - p) * self.pop[k2[idx]][self.ID_POS]
            else:
                pos_new = np.random.uniform(0, 1) * self.pop[k2[idx]][self.ID_POS]
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop_new)
