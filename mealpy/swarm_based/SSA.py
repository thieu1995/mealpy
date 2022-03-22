# !/usr/bin/env python
# Created by "Thieu" at 17:22, 29/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSSA(Optimizer):
    """
    My changed version of: Sparrow Search Algorithm (SSA)

    Notes
    ~~~~~
    + First, I sort the algorithm and find g-best and g-worst
    + In Eq. 4, Instead of using A+ and L, I used np.random.normal()
    + Some components (g_best_position, fitness updated) are missing in Algorithm 1 (paper)

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
        + PD (float): number of producers (percentage), default = 0.2
        + SD (float): number of sparrows who perceive the danger, default = 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SSA import BaseSSA
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
    >>> ST = 0.8
    >>> PD = 0.2
    >>> SD = 0.1
    >>> model = BaseSSA(problem_dict1, epoch, pop_size, ST, PD, SD)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J. and Shen, B., 2020. A novel swarm intelligence optimization approach:
    sparrow search algorithm. Systems Science & Control Engineering, 8(1), pp.22-34.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, ST=0.8, PD=0.2, SD=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
            PD (float): number of producers (percentage), default = 0.2
            SD (float): number of sparrows who perceive the danger, default = 0.1
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.ST = self.validator.check_float("ST", ST, (0, 1.0))
        self.PD = self.validator.check_float("PD", PD, (0, 1.0))
        self.SD = self.validator.check_float("SD", SD, (0, 1.0))
        self.n1 = int(self.PD * self.pop_size)
        self.n2 = int(self.SD * self.pop_size)
        self.nfe_per_epoch = 2 * self.pop_size - self.n2
        self.sort_flag = True

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
        return np.where(np.logical_and(lb <= position, position <= ub), position, np.random.uniform(lb, ub))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        r2 = np.random.uniform()  # R2 in [0, 1], the alarm value, random value
        pop_new = []
        for idx in range(0, self.pop_size):
            # Using equation (3) update the sparrow’s location;
            if idx < self.n1:
                if r2 < self.ST:
                    des = (epoch + 1) / (np.random.uniform() * self.epoch + self.EPSILON)
                    if des > 5:
                        des = np.random.normal()
                    x_new = self.pop[idx][self.ID_POS] * np.exp(des)
                else:
                    x_new = self.pop[idx][self.ID_POS] + np.random.normal() * np.ones(self.problem.n_dims)
            else:
                # Using equation (4) update the sparrow’s location;
                _, x_p, worst = self.get_special_solutions(self.pop, best=1, worst=1)
                g_best = x_p[0], g_worst = worst[0]
                if idx > int(self.pop_size / 2):
                    x_new = np.random.normal() * np.exp((g_worst[self.ID_POS] - self.pop[idx][self.ID_POS]) / (idx + 1) ** 2)
                else:
                    x_new = g_best[self.ID_POS] + np.abs(self.pop[idx][self.ID_POS] - g_best[self.ID_POS]) * np.random.normal()
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)
        pop_new, best, worst = self.get_special_solutions(pop_new, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop2 = deepcopy(pop_new[self.n2:])
        child = []
        for idx in range(0, len(pop2)):
            #  Using equation (5) update the sparrow’s location;
            if self.compare_agent(self.pop[idx], g_best):
                x_new = pop2[idx][self.ID_POS] + \
                        np.random.uniform(-1, 1) * (np.abs(pop2[idx][self.ID_POS] - g_worst[self.ID_POS]) /
                                                    (pop2[idx][self.ID_TAR][self.ID_FIT] - g_worst[self.ID_TAR][self.ID_FIT] + self.EPSILON))
            else:
                x_new = g_best[self.ID_POS] + np.random.normal() * np.abs(pop2[idx][self.ID_POS] - g_best[self.ID_POS])
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            child.append([pos_new, None])
        child = self.update_target_wrapper_population(child)
        child = self.greedy_selection_population(pop2, child)
        self.pop = pop_new[:self.n2] + child


class OriginalSSA(BaseSSA):
    """
    The original version of: Sparrow Search Algorithm (SSA)

    Links:
        1. https://doi.org/10.1080/21642583.2019.1708830

    Notes
    ~~~~~
    + The paper contains some unclear equations and symbol

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
        + PD (float): number of producers (percentage), default = 0.2
        + SD (float): number of sparrows who perceive the danger, default = 0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SSA import OriginalSSA
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
    >>> ST = 0.8
    >>> PD = 0.2
    >>> SD = 0.1
    >>> model = OriginalSSA(problem_dict1, epoch, pop_size, ST, PD, SD)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J. and Shen, B., 2020. A novel swarm intelligence optimization approach:
    sparrow search algorithm. Systems Science & Control Engineering, 8(1), pp.22-34.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, ST=0.8, PD=0.2, SD=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
            PD (float): number of producers (percentage), default = 0.2
            SD (float): number of sparrows who perceive the danger, default = 0.1
        """
        super().__init__(problem, epoch, pop_size, ST, PD, SD, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        r2 = np.random.uniform()  # R2 in [0, 1], the alarm value, random value
        pop_new = []
        for idx in range(0, self.pop_size):
            # Using equation (3) update the sparrow’s location;
            if idx < self.n1:
                if r2 < self.ST:
                    des = (idx + 1) / (np.random.uniform() * self.epoch + self.EPSILON)
                    if des > 5:
                        des = np.random.uniform()
                    x_new = self.pop[idx][self.ID_POS] * np.exp(des)
                else:
                    x_new = self.pop[idx][self.ID_POS] + np.random.normal() * np.ones(self.problem.n_dims)
            else:
                # Using equation (4) update the sparrow’s location;
                _, x_p, worst = self.get_special_solutions(self.pop, best=1, worst=1)
                g_best, g_worst = x_p[0], worst[0]
                if idx > int(self.pop_size / 2):
                    x_new = np.random.normal() * np.exp((g_worst[self.ID_POS] - self.pop[idx][self.ID_POS]) / (idx + 1) ** 2)
                else:
                    L = np.ones((1, self.problem.n_dims))
                    A = np.sign(np.random.uniform(-1, 1, (1, self.problem.n_dims)))
                    A1 = A.T * np.linalg.inv(np.matmul(A, A.T)) * L
                    x_new = g_best[self.ID_POS] + np.matmul(np.abs(self.pop[idx][self.ID_POS] - g_best[self.ID_POS]), A1)
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)
        pop_new, best, worst = self.get_special_solutions(pop_new, best=1, worst=1)
        g_best, g_worst = best[0], worst[0]
        pop2 = pop_new[self.n2:]
        child = []
        for idx in range(0, len(pop2)):
            #  Using equation (5) update the sparrow’s location;
            if self.compare_agent(self.pop[idx], g_best):
                x_new = pop2[idx][self.ID_POS] + \
                        np.random.uniform(-1, 1) * (np.abs(pop2[idx][self.ID_POS] - g_worst[self.ID_POS]) /
                                                    (pop2[idx][self.ID_TAR][self.ID_FIT] - g_worst[self.ID_TAR][self.ID_FIT] + self.EPSILON))
            else:
                x_new = g_best[self.ID_POS] + np.random.normal() * np.abs(pop2[idx][self.ID_POS] - g_best[self.ID_POS])
            pos_new = self.amend_position(x_new, self.problem.lb, self.problem.ub)
            child.append([pos_new, None])
        child = self.update_target_wrapper_population(child)
        child = self.greedy_selection_population(pop2, child)
        self.pop = pop_new[:self.n2] + child
