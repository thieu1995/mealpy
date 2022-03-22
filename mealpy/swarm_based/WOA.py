# !/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseWOA(Optimizer):
    """
    The original version of: Whale Optimization Algorithm (WOA)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2016.01.008

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.WOA import BaseWOA
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
    >>> model = BaseWOA(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S. and Lewis, A., 2016. The whale optimization algorithm.
    Advances in engineering software, 95, pp.51-67.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.create_solution(self.problem.lb, self.problem.ub)
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class HI_WOA(Optimizer):
    """
    The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)

    Links:
        1. https://ieenp.explore.ieee.org/document/8900003

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + feedback_max (int): maximum iterations of each feedback, default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.WOA import HI_WOA
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
    >>> feedback_max = 10
    >>> model = HI_WOA(problem_dict1, epoch, pop_size, feedback_max)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
    In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, feedback_max=10, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            feedback_max (int): maximum iterations of each feedback, default = 10
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.feedback_max = self.validator.check_int("feedback_max", feedback_max, [2, 2+int(self.epoch/2)])
        # The maximum of times g_best doesn't change -> need to change half of population
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        self.n_changes = int(pop_size / 2)
        ## Dynamic variable
        self.dyn_feedback_count = 0

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        pop_new = []
        for idx in range(0, self.pop_size):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = 0.5
            b = 1
            if np.random.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] - A * D
                else:
                    # x_rand = pop[np.random.np.random.randint(self.pop_size)]         # select random 1 position in pop
                    x_rand = self.create_solution(self.problem.lb, self.problem.ub)
                    D = np.abs(C * x_rand[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = x_rand[self.ID_POS] - A * D
            else:
                D1 = np.abs(self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        nfe_epoch += self.pop_size

        ## Feedback Mechanism
        _, current_best = self.get_global_best_solution(pop_new)
        if current_best[self.ID_TAR][self.ID_FIT] == self.g_best[self.ID_TAR][self.ID_FIT]:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0

        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = np.random.choice(range(0, self.pop_size), self.n_changes, replace=False)
            pop_child = self.create_population(self.n_changes)
            nfe_epoch += self.n_changes
            for idx_counter, idx in enumerate(idx_list):
                pop_new[idx] = pop_child[idx_counter]
        self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch
