# !/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from math import gamma
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseMSA(Optimizer):
    """
    My changed version of: Moth Search Algorithm (MSA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/59010-moth-search-ms-algorithm
        2. https://doi.org/10.1007/s12293-016-0212-3

    Notes
    ~~~~~
    + The matlab version of original paper is not good (especially convergence chart)
    + I add Normal random number (Gaussian distribution) in each updating equation (Better performance)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_best (int): [3, 10], how many of the best moths to keep from one generation to the next, default=5
        + partition (float): [0.3, 0.8], The proportional of first partition, default=0.5
        + max_step_size (float): [0.5, 2.0], Max step size used in Levy-flight technique, default=1.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MSA import BaseMSA
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
    >>> n_best = 5
    >>> partition = 0.5
    >>> max_step_size = 1.0
    >>> model = BaseMSA(problem_dict1, epoch, pop_size, n_best, partition, max_step_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., 2018. Moth search algorithm: a bio-inspired metaheuristic algorithm for
    global optimization problems. Memetic Computing, 10(2), pp.151-164.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, n_best=5, partition=0.5, max_step_size=1.0, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_best (int): how many of the best moths to keep from one generation to the next, default=5
            partition (float): The proportional of first partition, default=0.5
            max_step_size (float): Max step size used in Levy-flight technique, default=1.0
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size/2)])
        self.partition = self.validator.check_float("partition", partition, (0, 1.0))
        self.max_step_size = self.validator.check_float("max_step_size", max_step_size, (0, 5.0))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        # np1 in paper
        self.n_moth1 = int(np.ceil(self.partition * self.pop_size))
        # np2 in paper, we actually don't need this variable
        self.n_moth2 = self.pop_size - self.n_moth1
        # you can change this ratio so as to get much better performance
        self.golden_ratio = (np.sqrt(5) - 1) / 2.0

    def _levy_walk(self, iteration):
        beta = 1.5  # Eq. 2.23
        sigma = (gamma(1 + beta) * np.sin(np.pi * (beta - 1) / 2) / (gamma(beta / 2) * (beta - 1) * 2 ** ((beta - 2) / 2))) ** (1 / (beta - 1))
        u = np.random.uniform(self.problem.lb, self.problem.ub) * sigma
        v = np.random.uniform(self.problem.lb, self.problem.ub)
        step = u / np.abs(v) ** (1.0 / (beta - 1))  # Eq. 2.21
        scale = self.max_step_size / (iteration + 1)
        delta_x = scale * step
        return delta_x

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_best = deepcopy(self.pop[:self.n_best])

        pop_new = []
        for idx in range(0, self.pop_size):
            # Migration operator
            if idx < self.n_moth1:
                # scale = self.max_step_size / (epoch+1)       # Smaller step for local walk
                pos_new = self.pop[idx][self.ID_POS] + np.random.normal() * self._levy_walk(epoch)
            else:
                # Flying in a straight line
                temp_case1 = self.pop[idx][self.ID_POS] + np.random.normal() * \
                             self.golden_ratio * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                temp_case2 = self.pop[idx][self.ID_POS] + np.random.normal() * \
                             (1.0 / self.golden_ratio) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = np.where(np.random.uniform(self.problem.n_dims) < 0.5, temp_case2, temp_case1)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop, _ = self.get_global_best_solution(self.pop)
        # Replace the worst with the previous generation's elites.
        for i in range(0, self.n_best):
            self.pop[-1 - i] = deepcopy(pop_best[i])
