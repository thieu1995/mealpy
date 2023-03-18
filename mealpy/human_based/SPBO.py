#!/usr/bin/env python
# Created by "Thieu" at 17:19, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSPBO(Optimizer):
    """
    The original version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Weak algorithm
        2. Consume too much time because of ndim * pop_size updating times.

    Links:
       1. https://www.sciencedirect.com/science/article/abs/pii/S0965997820301484
       2. https://www.mathworks.com/matlabcentral/fileexchange/80991-student-psycology-based-optimization-spbo-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.SPBO import OriginalSPBO
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
    >>> model = OriginalSPBO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Das, B., Mukherjee, V., & Das, D. (2020). Student psychology based optimization algorithm: A new population based
    optimization algorithm for solving optimization problems. Advances in Engineering software, 146, 102804.
    """
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        for jdx in range(0, self.problem.n_dims):
            idx_best = self.get_index_best(self.pop)
            mid = np.random.randint(1, self.pop_size-1)
            x_mean = np.mean([agent[self.ID_POS] for agent in self.pop], axis=0)
            pop_new = []
            for idx in range(0, self.pop_size):
                if idx == idx_best:
                    k = np.random.choice([1, 2])
                    j = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                    new_pos = self.g_best[self.ID_POS] + (-1)**k * np.random.rand() * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
                elif idx < mid:
                    ## Good Student
                    if np.random.rand() > np.random.rand():
                        new_pos = self.g_best[self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        new_pos = self.pop[idx][self.ID_POS] + np.random.rand() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                            np.random.rand() * (self.pop[idx][self.ID_POS] - x_mean)
                else:
                    ## Average Student
                    if np.random.rand() > np.random.rand():
                        new_pos = self.pop[idx][self.ID_POS] + np.random.rand() * (x_mean - self.pop[idx][self.ID_POS])
                    else:
                        new_pos = self.generate_position(self.problem.lb, self.problem.ub)
                new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
                pop_new.append([new_pos, None])
                if self.mode not in self.AVAILABLE_MODES:
                    new_tar = self.get_target_wrapper(new_pos)
                    self.pop[idx] = self.get_better_solution([new_pos, new_tar], self.pop[idx])
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_wrapper_population(pop_new)
                self.pop = self.greedy_selection_population(self.pop, pop_new)


class DevSPBO(OriginalSPBO):
    """
    The developed version of: Student Psychology Based Optimization (SPBO)

    Notes:
        1. Replace random number by normal random number
        2. Sort the population and select 1/3 pop size for each category

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.SPBO import DevSPBO
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
    >>> model = DevSPBO(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """
    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        good = int(self.pop_size / 3)
        average = 2 * int(self.pop_size / 3)
        x_mean = np.mean([agent[self.ID_POS] for agent in self.pop], axis=0)
        pop_new = []
        for idx in range(0, self.pop_size):
            if idx == 0:
                j = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
                new_pos = self.g_best[self.ID_POS] + np.random.random(self.problem.n_dims) * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
            elif idx < good:    ## Good Student
                if np.random.rand() > np.random.rand():
                    new_pos = self.g_best[self.ID_POS] + np.random.normal(0, 1) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    ra = np.random.rand()
                    new_pos = self.pop[idx][self.ID_POS] + ra * \
                              (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + (1 - ra) * (self.pop[idx][self.ID_POS] - x_mean)
            elif idx < average:  ## Average Student
                new_pos = self.pop[idx][self.ID_POS] + np.random.normal(0, 1) * (x_mean - self.pop[idx][self.ID_POS])
            else:
                new_pos = self.generate_position(self.problem.lb, self.problem.ub)
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            pop_new.append([new_pos, None])
            if self.mode not in self.AVAILABLE_MODES:
                new_tar = self.get_target_wrapper(new_pos)
                self.pop[idx] = self.get_better_solution([new_pos, new_tar], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
