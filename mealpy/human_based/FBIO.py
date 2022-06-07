# !/usr/bin/env python
# Created by "Thieu" at 08:57, 14/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseFBIO(Optimizer):
    """
    My changed version of: Forensic-Based Investigation Optimization (FBIO)

    Notes
    ~~~~~
    I remove all the third loop, change a few equations and the flow of the algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.FBIO import BaseFBIO
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
    >>> model = BaseFBIO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
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
        self.nfe_per_epoch = 4 * self.pop_size
        self.sort_flag = False

    def probability__(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = np.max(list_fitness)
        min1 = np.min(list_fitness)
        return (max1 - list_fitness) / (max1 - min1 + self.EPSILON)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Investigation team - team A
        # Step A1
        pop_new = []
        for idx in range(0, self.pop_size):
            n_change = np.random.randint(0, self.problem.n_dims)
            nb1, nb2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = deepcopy(self.pop[idx][self.ID_POS])
            pos_a[n_change] = self.pop[idx][self.ID_POS][n_change] + np.random.normal() * \
                (self.pop[idx][self.ID_POS][n_change] - (self.pop[nb1][self.ID_POS][n_change] + self.pop[nb2][self.ID_POS][n_change]) / 2)
            pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            pop_new.append([pos_a, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_a)
                self.pop[idx] = self.get_better_solution([pos_a, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
        list_fitness = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        prob = self.probability__(list_fitness)

        # Step A2
        pop_child = []
        for idx in range(0, self.pop_size):
            if np.random.rand() > prob[idx]:
                r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
                temp = self.g_best[self.ID_POS] + self.pop[r1][self.ID_POS] + np.random.uniform() * (self.pop[r2][self.ID_POS] - self.pop[r3][self.ID_POS])
                condition = np.random.random(self.problem.n_dims) < 0.5
                pos_new = np.where(condition, temp, self.pop[idx][self.ID_POS])
            else:
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_child.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_new)
                self.pop[idx] = self.get_better_solution([pos_new, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)

        ## Persuing team - team B
        ## Step B1
        pop_new = []
        for idx in range(0, self.pop_size):
            ### Remove third loop here also
            ### Eq.(6) in FBI Inspired Meta-Optimization
            pos_b = np.random.uniform(0, 1, self.problem.n_dims) * self.pop[idx][self.ID_POS] + \
                    np.random.uniform(0, 1, self.problem.n_dims) * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_new.append([pos_b, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Step B2
        pop_child = []
        for idx in range(0, self.pop_size):
            rr = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            if self.compare_agent(self.pop[idx], self.pop[rr]):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                        (self.pop[rr][self.ID_POS] - self.pop[idx][self.ID_POS]) + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[rr][self.ID_POS])
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                        (self.pop[idx][self.ID_POS] - self.pop[rr][self.ID_POS]) + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_child.append([pos_b, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)


class OriginalFBIO(BaseFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)

    Links:
        1. https://doi.org/10.1016/j.asoc.2020.106339
        2. https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.FBIO import OriginalFBIO
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
    >>> model = OriginalFBIO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Chou, J.S. and Nguyen, N.M., 2020. FBI inspired meta-optimization. Applied Soft Computing, 93, p.106339.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = 4 * self.pop_size
        self.sort_flag = False

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        rand_pos = np.random.uniform(lb, ub)
        condition = np.logical_and(lb <= position, position <= ub)
        return np.where(condition, position, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Investigation team - team A
        # Step A1
        pop_new = []
        for idx in range(0, self.pop_size):
            n_change = np.random.randint(0, self.problem.n_dims)
            nb1, nb2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = deepcopy(self.pop[idx][self.ID_POS])
            pos_a[n_change] = self.pop[idx][self.ID_POS][n_change] + (np.random.uniform() - 0.5) * 2 * \
                (self.pop[idx][self.ID_POS][n_change] - (self.pop[nb1][self.ID_POS][n_change] + self.pop[nb2][self.ID_POS][n_change]) / 2)
            ## Not good move here, change only 1 variable but check bound of all variable in solution
            pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            pop_new.append([pos_a, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_a)
                self.pop[idx] = self.get_better_solution([pos_a, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        # Step A2
        list_fitness = np.array([item[self.ID_TAR][self.ID_FIT] for item in self.pop])
        prob = self.probability__(list_fitness)
        pop_child = []
        for idx in range(0, self.pop_size):
            if np.random.uniform() > prob[idx]:
                r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_a = deepcopy(self.pop[idx][self.ID_POS])
                Rnd = np.floor(np.random.uniform() * self.problem.n_dims) + 1

                for j in range(0, self.problem.n_dims):
                    if (np.random.uniform() < np.random.uniform() or Rnd == j):
                        pos_a[j] = self.g_best[self.ID_POS][j] + self.pop[r1][self.ID_POS][j] + \
                                   np.random.uniform() * (self.pop[r2][self.ID_POS][j] - self.pop[r3][self.ID_POS][j])
                    ## In the original matlab code they do the else condition here, not good again because no need else here
                ## Same here, they do check the bound of all variable in solution
                ## pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            else:
                pos_a = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_a = self.amend_position(pos_a, self.problem.lb, self.problem.ub)
            pop_child.append([pos_a, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_a)
                self.pop[idx] = self.get_better_solution([pos_a, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)

        ## Persuing team - team B
        ## Step B1
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_b = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b[j] = np.random.uniform() * self.pop[idx][self.ID_POS][j] + \
                           np.random.uniform() * (self.g_best[self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_new.append([pos_b, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)

        ## Step B2
        pop_child = []
        for idx in range(0, self.pop_size):
            ### Not good move here again
            rr = np.random.randint(0, self.pop_size)
            while rr == idx:
                rr = np.random.randint(0, self.pop_size)
            if self.compare_agent(self.pop[idx], self.pop[rr]):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (self.pop[rr][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                        np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[rr][self.ID_POS])
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (self.pop[idx][self.ID_POS] - self.pop[rr][self.ID_POS]) + \
                        np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
            pos_b = self.amend_position(pos_b, self.problem.lb, self.problem.ub)
            pop_child.append([pos_b, None])
            if self.mode not in self.AVAILABLE_MODES:
                target = self.get_target_wrapper(pos_b)
                self.pop[idx] = self.get_better_solution([pos_b, target], self.pop[idx])
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_wrapper_population(pop_child)
            self.pop = self.greedy_selection_population(pop_child, self.pop)
