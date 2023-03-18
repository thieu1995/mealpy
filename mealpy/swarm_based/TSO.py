#!/usr/bin/env python
# Created by "Thieu" at 17:52, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTSO(Optimizer):
    """
    The original version of: Tuna Swarm Optimization (TSO)

    Links:
        1. https://www.hindawi.com/journals/cin/2021/9210050/
        2. https://www.mathworks.com/matlabcentral/fileexchange/101734-tuna-swarm-optimization

    Notes:
        1. Two variables that authors consider it as a constants (aa = 0.7 and zz = 0.05)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.TSO import OriginalTSO
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
    >>> model = OriginalTSO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Xie, L., Han, T., Zhou, H., Zhang, Z. R., Han, B., & Tang, A. (2021). Tuna swarm optimization: a novel swarm-based
    metaheuristic algorithm for global optimization. Computational intelligence and Neuroscience, 2021.
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
        self.sort_flag = True

    def initialize_variables(self):
        self.aa = 0.7
        self.zz = 0.05

    def get_new_local_pos__(self, C, a1, a2, t, epoch):
        if np.random.rand() < self.zz:
            local_pos = self.generate_position(self.problem.lb, self.problem.ub)
        else:
            if np.random.rand() < 0.5:
                r1 = np.random.rand()
                beta = np.exp(r1 * np.exp(3*np.cos(np.pi*((self.epoch - epoch) / self.epoch)))) * np.cos(2*np.pi*r1)
                if np.random.rand() < C:
                    local_pos = a1*(self.g_best[self.ID_POS] + beta * np.abs(self.g_best[self.ID_POS] - self.pop[0][self.ID_POS])) + \
                        a2 * self.pop[0][self.ID_POS]       # Equation (8.3)
                else:
                    rand_pos = self.generate_position(self.problem.lb, self.problem.ub)
                    local_pos = a1 * (rand_pos + beta*np.abs(rand_pos - self.pop[0][self.ID_POS])) + a2 * self.pop[0][self.ID_POS]  # Equation (8.1)
            else:
                tf = np.random.choice([-1, 1])
                if np.random.rand() < 0.5:
                    local_pos = tf * t**2 * self.pop[0][self.ID_POS]        # Eq 9.2
                else:
                    local_pos = self.g_best[self.ID_POS] + np.random.rand(self.problem.n_dims) * (self.g_best[self.ID_POS] - self.pop[0][self.ID_POS]) + \
                        tf * t**2 * (self.g_best[self.ID_POS] - self.pop[0][self.ID_POS])
        return local_pos

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        C = (epoch + 1) / self.epoch
        a1 = self.aa + (1 - self.aa) * C
        a2 = (1 - self.aa) - (1 - self.aa) * C
        t = (1 - (epoch+1) / self.epoch) ** ((epoch+1) / self.epoch)

        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                pos_new = self.get_new_local_pos__(C, a1, a2, t, epoch)
            else:
                if np.random.rand() < self.zz:
                    pos_new = self.generate_position(self.problem.lb, self.problem.ub)
                else:
                    if np.random.rand() > 0.5:
                        r1 = np.random.rand()
                        beta = np.exp(r1 * np.exp(3*np.cos(np.pi * (self.epoch - epoch)/self.epoch))) * np.cos(2*np.pi*r1)
                        if np.random.rand() < C:
                            pos_new = a1 * (self.g_best[self.ID_POS] + beta*np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])) + \
                                a2 * self.pop[idx-1][self.ID_POS]       # Eq. 8.4
                        else:
                            rand_pos = self.generate_position(self.problem.lb, self.problem.ub)
                            pos_new = a1 * (rand_pos + beta*np.abs(rand_pos - self.pop[idx][self.ID_POS])) + a2 * self.pop[idx-1][self.ID_POS]  # Eq 8.2
                    else:
                        tf = np.random.choice([-1, 1])
                        if np.random.rand() < 0.5:
                            pos_new = self.g_best[self.ID_POS] + \
                                      np.random.rand(self.problem.n_dims) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) +\
                                      tf * t**2 * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])       # Eq 9.1
                        else:
                            pos_new = tf * t**2 * self.pop[idx][self.ID_POS]        # Eq 9.2
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop_new)
