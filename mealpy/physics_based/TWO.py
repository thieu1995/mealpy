# !/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseTWO(Optimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import BaseTWO
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
    >>> model = BaseTWO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_WEIGHT = 2

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
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def create_solution(self, lb=None, ub=None):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, target, weight]
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        weight = 0.0
        return [position, target, weight]

    def _update_weight(self, teams):
        _, best, worst = self.get_special_solutions(teams, best=1, worst=1)
        best_fit = best[0][self.ID_TAR][self.ID_FIT]
        worst_fit = worst[0][self.ID_TAR][self.ID_FIT]
        if best_fit == worst_fit:
            for i in range(self.pop_size):
                teams[i][self.ID_WEIGHT] = np.random.uniform(0.5, 1.5)
        else:
            for i in range(self.pop_size):
                teams[i][self.ID_WEIGHT] = (teams[i][self.ID_TAR][self.ID_FIT] - worst_fit) / (best_fit - worst_fit + self.EPSILON) + 1
        return teams

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        self.pop = self._update_weight(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = pop_new[i][self.ID_POS].astype(float)
            for j in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pos_new += delta_x
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        for i in range(self.pop_size):
            pos_new = pop_new[i][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)


class OppoTWO(BaseTWO):
    """
    The opossition-based learning version of: Tug of War Optimization (OTWO)

    Notes
    ~~~~~
    + Applied the idea of Opposition-based learning technique

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import OppoTWO
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
    >>> model = OppoTWO(problem_dict1, epoch, pop_size)
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
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialization(self):
        pop_temp = self.create_population(int(self.pop_size / 2))
        pop_oppo = []
        for i in range(len(pop_temp)):
            pos_opposite = self.problem.ub + self.problem.lb - pop_temp[i][self.ID_POS]
            pos_opposite = self.amend_position(pos_opposite, self.problem.lb, self.problem.ub)
            pop_oppo.append([pos_opposite, None, 0.0])
        pop_oppo = self.update_target_wrapper_population(pop_oppo)
        self.pop = pop_temp + pop_oppo
        self.pop = self._update_weight(self.pop)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Apply force of others solution on each individual solution
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = pop_new[i][self.ID_POS].astype(float)
            for j in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[j][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[j][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[j][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pos_new += delta_x
            self.pop[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

        ## Amend solution and update fitness value
        for i in range(self.pop_size):
            pos_new = self.g_best[self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) / (epoch + 1) * \
                      (self.g_best[self.ID_POS] - pop_new[i][self.ID_POS])
            conditions = np.logical_or(pop_new[i][self.ID_POS] < self.problem.lb, pop_new[i][self.ID_POS] > self.problem.ub)
            conditions = np.logical_and(conditions, np.random.uniform(0, 1, self.problem.n_dims) < 0.5)
            pos_new = np.where(conditions, pos_new, self.pop[i][self.ID_POS])
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        pop_new = self.update_target_wrapper_population(pop_new)

        ## Opposition-based here
        for i in range(self.pop_size):
            if self.compare_agent(pop_new[i], self.pop[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                C_op = self.create_opposition_position(self.pop[i][self.ID_POS], self.g_best[self.ID_POS])
                C_op = self.amend_position(C_op, self.problem.lb, self.problem.ub)
                target_op = self.get_target_wrapper(C_op)
                if self.compare_agent([C_op, target_op], self.pop[i]):
                    self.pop[i] = [C_op, target_op, 0.0]
        self.pop = self._update_weight(self.pop)


class LevyTWO(BaseTWO):
    """
    The Levy-flight version of: Tug of War Optimization (LTWO)

    Notes
    ~~~~~
    + Applied the idea of Levy-flight technique

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import LevyTWO
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
    >>> model = LevyTWO(problem_dict1, epoch, pop_size)
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
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pos_new +=delta_x
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)

        ### Apply levy-flight here
        for i in range(self.pop_size):
            if self.compare_agent(pop_new[i], self.pop[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                levy_step = self.get_levy_flight_step(beta=1.0, multiplier=10, case=-1)
                pos_new = pop_new[i][self.ID_POS] + np.sign(np.random.random() - 0.5) * levy_step
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                self.pop[i] = [pos_new, target, 0.0]
        self.pop = self._update_weight(pop_new)


class EnhancedTWO(OppoTWO, LevyTWO):
    """
    The original version of: Enhenced Tug of War Optimization (ETWO)

    Links:
        1. https://doi.org/10.1016/j.procs.2020.03.063

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.TWO import EnhancedTWO
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
    >>> r_rate = 0.3
    >>> ps_rate = 0.85
    >>> p_field = 0.1
    >>> n_field = 0.45
    >>> model = EnhancedTWO(problem_dict1, epoch, pop_size, r_rate, ps_rate, p_field, n_field)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Hoang, B., Nguyen, G. and Nguyen, B.M., 2020. A new workload prediction model using
    extreme learning machine and enhanced tug of war optimization. Procedia Computer Science, 170, pp.362-369.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialization(self):
        pop_temp = self.create_population(self.pop_size)
        pop_oppo = deepcopy(pop_temp)
        for i in range(self.pop_size):
            pos_opposite = self.problem.ub + self.problem.lb - pop_temp[i][self.ID_POS]
            pop_oppo[i][self.ID_POS] = self.amend_position(pos_opposite, self.problem.lb, self.problem.ub)
        pop_oppo = self.update_target_wrapper_population(pop_oppo)
        self.pop = self.get_sorted_strim_population(pop_temp + pop_oppo, self.pop_size)
        self.pop = self._update_weight(self.pop)
        self.g_best = deepcopy(self.pop[0])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = deepcopy(self.pop)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for k in range(self.pop_size):
                if self.pop[i][self.ID_WEIGHT] < self.pop[k][self.ID_WEIGHT]:
                    force = max(self.pop[i][self.ID_WEIGHT] * self.muy_s, self.pop[k][self.ID_WEIGHT] * self.muy_s)
                    resultant_force = force - self.pop[i][self.ID_WEIGHT] * self.muy_k
                    g = self.pop[k][self.ID_POS] - self.pop[i][self.ID_POS]
                    acceleration = resultant_force * g / (self.pop[i][self.ID_WEIGHT] * self.muy_k)
                    delta_x = 1 / 2 * acceleration + np.power(self.alpha, epoch + 1) * self.beta * \
                              (self.problem.ub - self.problem.lb) * np.random.randn(self.problem.n_dims)
                    pos_new += delta_x
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        for i in range(self.pop_size):
            pos_new = self.pop[i][self.ID_POS].astype(float)
            for j in range(self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                    if np.random.random() <= 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] + np.random.randn() / (epoch + 1) * \
                                                     (self.g_best[self.ID_POS][j] - pos_new[j])
                        if pos_new[j] < self.problem.lb[j] or pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.pop[i][self.ID_POS][j]
                    else:
                        if pos_new[j] < self.problem.lb[j]:
                            pos_new[j] = self.problem.lb[j]
                        if pos_new[j] > self.problem.ub[j]:
                            pos_new[j] = self.problem.ub[j]
            pop_new[i][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        pop_new = self.update_target_wrapper_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)
        self.pop = self._update_weight(pop_new)

        for i in range(self.pop_size):
            if self.compare_agent(pop_new[i], self.pop[i]):
                self.pop[i] = deepcopy(pop_new[i])
            else:
                C_op = self.create_opposition_position(self.pop[i][self.ID_POS], self.g_best[self.ID_POS])
                C_op = self.amend_position(C_op, self.problem.lb, self.problem.ub)
                target_op = self.get_target_wrapper(C_op)
                if self.compare_agent([C_op, target_op], self.pop[i]):
                    self.pop[i] = [C_op, target_op, 0.0]
                else:
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                    pos_new = pop_new[i][self.ID_POS] + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random() - 0.5) * levy_step
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    target = self.get_target_wrapper(pos_new)
                    self.pop[i] = [pos_new, target, 0.0]
        self.pop = self._update_weight(pop_new)
