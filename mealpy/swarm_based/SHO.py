#!/usr/bin/env python
# Created by "Thieu" at 10:55, 02/12/2019 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSHO(Optimizer):
    """
    The original version of: Spotted Hyena Optimizer (SHO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.05.014

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
        + N_tried (int): default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SHO import OriginalSHO
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
    >>> h_factor = 5.0
    >>> N_tried = 10
    >>> model = OriginalSHO(epoch, pop_size, h_factor, N_tried)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
    technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """

    def __init__(self, epoch=10000, pop_size=100, h_factor=5., N_tried=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h_factor (float): default = 5, coefficient linearly decreased from 5.0 to 0
            N_tried (int): default = 10,
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.h_factor = self.validator.check_float("h_factor", h_factor, (0.5, 10.0))
        self.N_tried = self.validator.check_int("N_tried", N_tried, (1, float("inf")))
        self.set_parameters(["epoch", "pop_size", "h_factor", "N_tried"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            h = self.h_factor - (epoch + 1.0) * (self.h_factor / self.epoch)
            rd1 = np.random.uniform(0, 1, self.problem.n_dims)
            rd2 = np.random.uniform(0, 1, self.problem.n_dims)
            B = 2 * rd1
            E = 2 * h * rd2 - h

            if np.random.rand() < 0.5:
                D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] - np.dot(E, D_h)
            else:
                N = 1
                for i in range(0, self.N_tried):
                    pos_temp = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * \
                              np.random.uniform(self.problem.lb, self.problem.ub)
                    pos_temp = self.amend_position(pos_temp, self.problem.lb, self.problem.ub)
                    target = self.get_target_wrapper(pos_temp)
                    if self.compare_agent([pos_temp, target], self.g_best):
                        N += 1
                        break
                    N += 1
                circle_list = []
                idx_list = np.random.choice(range(0, self.pop_size), N, replace=False)
                for j in range(0, N):
                    D_h = np.abs(np.dot(B, self.g_best[self.ID_POS]) - self.pop[idx_list[j]][self.ID_POS])
                    p_k = self.g_best[self.ID_POS] - np.dot(E, D_h)
                    circle_list.append(p_k)
                pos_new = np.mean(np.array(circle_list), axis=0)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
