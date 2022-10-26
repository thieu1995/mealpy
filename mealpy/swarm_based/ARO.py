#!/usr/bin/env python
# Created by "Thieu" at 22:46, 26/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalARO(Optimizer):
    """
    The original version of: Artificial Rabbits Optimization (ARO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2022.105082
        2. https://www.mathworks.com/matlabcentral/fileexchange/110250-artificial-rabbits-optimization-aro

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.ARO import OriginalARO
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
    >>> model = OriginalARO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, L., Cao, Q., Zhang, Z., Mirjalili, S., & Zhao, W. (2022). Artificial rabbits optimization: A new bio-inspired
    meta-heuristic algorithm for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 114, 105082.
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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        theta = 2 * (1 - (epoch+1)/self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * (np.sin(2*np.pi*np.random.rand()))
            temp = np.zeros(self.problem.n_dims)
            rd_index = np.random.choice(np.arange(0, self.problem.n_dims), int(np.ceil(np.random.rand()*self.problem.n_dims)), replace=False)
            temp[rd_index] = 1
            R = L * temp        # Eq 2
            A = 2 * np.log(1.0 / np.random.rand()) * theta      # Eq. 15
            if A > 1:
                rand_idx = np.random.randint(0, self.pop_size)
                pos_new = self.pop[rand_idx][self.ID_POS] + R * (self.pop[idx][self.ID_POS] - self.pop[rand_idx][self.ID_POS]) + \
                    np.round(0.5 * (0.05 + np.random.rand())) * np.random.normal(0, 1)      # Eq. 1
            else:
                gr = np.zeros(self.problem.n_dims)
                rd_index = np.random.choice(np.arange(0, self.problem.n_dims), int(np.ceil(np.random.rand() * self.problem.n_dims)), replace=False)
                gr[rd_index] = 1        # Eq. 12
                H = np.random.normal(0, 1) * (epoch / self.epoch)       # Eq. 8
                b = self.pop[idx][self.ID_POS] + H * gr * self.pop[idx][self.ID_POS]        # Eq. 13
                pos_new = self.pop[idx][self.ID_POS] + R * (np.random.rand() * b - self.pop[idx][self.ID_POS])      # Eq. 11
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
