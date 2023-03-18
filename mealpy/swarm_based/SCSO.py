#!/usr/bin/env python
# Created by "Thieu" at 17:36, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSCSO(Optimizer):
    """
    The original version of: Sand Cat Swarm Optimization (SCSO)

    Links:
        1. https://link.springer.com/article/10.1007/s00366-022-01604-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/110185-sand-cat-swarm-optimization

    Notes:
        1. The matlab code will not work since the R value always in the range (-1, 1).
        2. The authors make a mistake in matlab code. It should be 0 <= R <= 1 in the If condition

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SCSO import OriginalSCSO
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
    >>> model = OriginalSCSO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Seyyedabbasi, A., & Kiani, F. (2022). Sand Cat swarm optimization: a nature-inspired algorithm to
    solve global optimization problems. Engineering with Computers, 1-25.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.P = np.arange(1, 361)
        self.sort_flag = False

    def initialize_variables(self):
        self.S = 2      # maximum Sensitivity range

    def get_index_roulette_wheel_selection__(self, p):
        p = p / np.sum(p)
        c = np.cumsum(p)
        return np.argwhere(np.random.rand() < c)[0][0]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        t = self.epoch + 1
        guides_r = self.S - (self.S * t / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand() * guides_r
            R = (2*guides_r)*np.random.rand() - guides_r        # controls to transition phases
            pos_new = self.pop[idx][self.ID_POS].copy()
            for jdx in range(0, self.problem.n_dims):
                teta = self.get_index_roulette_wheel_selection__(self.P)
                if 0 <= R <= 1:
                    rand_pos = np.abs(np.random.rand() * self.g_best[self.ID_POS][jdx] - self.pop[idx][self.ID_POS][jdx])
                    pos_new[jdx] = self.g_best[self.ID_POS][jdx] - r * rand_pos * np.cos(teta)
                else:
                    cp = int(np.random.rand() * self.pop_size)
                    pos_new[jdx] = r * (self.pop[cp][self.ID_POS][jdx] - np.random.rand() * self.pop[idx][self.ID_POS][jdx])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop_new)
