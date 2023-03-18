#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBBOA(Optimizer):
    """
    The original version of: Brown-Bear Optimization Algorithm (BBOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/125490-brown-bear-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.BBOA import OriginalBBOA
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
    >>> model = OriginalBBOA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Prakash, T., Singh, P. P., Singh, V. P., & Singh, S. N. (2023). A Novel Brown-bear Optimization
    Algorithm for Solving Economic Dispatch Problem. In Advanced Control & Optimization Paradigms for
    Energy System Operation and Management (pp. 137-164). River Publishers.
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
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pp = (epoch+1) / self.epoch

        ## Pedal marking behaviour
        pop_new = []
        for idx in range(0, self.pop_size):
            if pp <= (epoch+1)/3:           # Gait while walking
                pos_new = self.pop[idx][self.ID_POS] + (-pp * np.random.rand(self.problem.n_dims) * self.pop[idx][self.ID_POS])
            elif (epoch+1)/3 < pp <= 2*(epoch+1)/3:     # Careful Stepping
                qq = pp * np.random.rand(self.problem.n_dims)
                pos_new = self.pop[idx][self.ID_POS] + (qq * (self.g_best[self.ID_POS] - np.random.randint(1, 3) * self.g_worst[self.ID_POS]))
            else:
                ww = 2 * pp * np.pi * np.random.rand(self.problem.n_dims)
                pos_new = self.pop[idx][self.ID_POS] + (ww*self.g_best[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS])) - (ww*self.g_worst[self.ID_POS] - np.abs(self.pop[idx][self.ID_POS]))
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Sniffing of pedal marks
        pop_new = []
        for idx in range(0, self.pop_size):
            kk = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_agent(self.pop[idx], self.pop[kk]):
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[idx][self.ID_POS] - self.pop[kk][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] + np.random.rand() * (self.pop[kk][self.ID_POS] - self.pop[idx][self.ID_POS])
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
