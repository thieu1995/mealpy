#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMGO(Optimizer):
    """
    The original version of: Mountain Gazelle Optimizer (MGO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0965997822001831
        2. https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MGO import OriginalMGO
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
    >>> model = OriginalMGO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., Khodadadi, N., & Mirjalili, S. (2022). Mountain gazelle optimizer: a new
    nature-inspired metaheuristic algorithm for global optimization problems. Advances in Engineering Software, 174, 103282.
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
        self.sort_flag = True

    def coefficient_vector__(self, n_dims, epoch, max_epoch):
        a2 = -1 + epoch * ((-1) / max_epoch)
        u = np.random.randn(n_dims)
        v = np.random.randn(n_dims)
        cofi = np.zeros((4, n_dims))
        cofi[0, :] = np.random.rand(n_dims)
        cofi[1, :] = (a2 + 1) + np.random.rand()
        cofi[2, :] = a2 * np.random.randn(n_dims)
        cofi[3, :] = u * np.power(v, 2) * np.cos((np.random.rand() * 2) * u)
        return cofi

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            idxs_rand = np.random.permutation(self.pop_size)[:int(np.ceil(self.pop_size/3))]
            pos_list = np.array([ self.pop[mm][self.ID_POS] for mm in idxs_rand ])
            idx_rand = np.random.randint(int(np.ceil(self.pop_size / 3)), self.pop_size)
            M = self.pop[idx_rand][self.ID_POS] * np.floor(np.random.normal()) + np.mean(pos_list, axis=0) * np.ceil(np.random.normal())

            # Calculate the vector of coefficients
            cofi = self.coefficient_vector__(self.problem.n_dims, epoch+1, self.epoch)
            A = np.random.randn(self.problem.n_dims) * np.exp(2 - (epoch+1) * (2. / self.epoch))
            D = (np.abs(self.pop[idx][self.ID_POS]) + np.abs(self.g_best[self.ID_POS]))*(2 * np.random.rand() - 1)

            # Update the location
            x2 = self.g_best[self.ID_POS] - np.abs((np.random.randint(1, 3)*M - np.random.randint(1, 3)*self.pop[idx][self.ID_POS]) * A) * cofi[np.random.randint(0, 4), :]
            x3 = M + cofi[np.random.randint(0, 4), :] + (np.random.randint(1, 3)*self.g_best[self.ID_POS] - np.random.randint(1, 3)*self.pop[np.random.randint(self.pop_size)][self.ID_POS])*cofi[np.random.randint(0, 4), :]
            x4 = self.pop[idx][self.ID_POS] - D + (np.random.randint(1, 3)*self.g_best[self.ID_POS] - np.random.randint(1, 3)*M) * cofi[np.random.randint(0, 4), :]

            x1 = self.generate_position(self.problem.lb, self.problem.ub)
            x1 = self.amend_position(x1, self.problem.lb, self.problem.ub)
            x2 = self.amend_position(x2, self.problem.lb, self.problem.ub)
            x3 = self.amend_position(x3, self.problem.lb, self.problem.ub)
            x4 = self.amend_position(x4, self.problem.lb, self.problem.ub)

            pop_new += [[x1, None], [x2, None], [x3, None], [x4, None]]
            if self.mode not in self.AVAILABLE_MODES:
                for jdx in range(-4, 0):
                    pop_new[jdx][self.ID_TAR] = self.get_target_wrapper(pop_new[jdx][self.ID_POS])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.get_sorted_strim_population(self.pop + pop_new, self.pop_size)
