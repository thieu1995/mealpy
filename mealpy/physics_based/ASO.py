# !/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseASO(Optimizer):
    """
    The original version of: Atom Search Optimization (ASO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2018.08.030
        2. https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + alpha (int): Depth weight, default = 50
        + beta (float): Multiplier weight, default = 0.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.physics_based.ASO import BaseASO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> alpha = 50
    >>> beta = 0.2
    >>> model = BaseASO(problem_dict1, epoch, pop_size, alpha, beta)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2019. Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem. Knowledge-Based Systems, 163, pp.283-304.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_VEL = 2  # Velocity
    ID_MAS = 3  # Mass of atom

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=50, beta=0.2, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): [10, 100], Depth weight, default = 50
            beta (float): [0.1, 1.0], Multiplier weight, default = 0.2
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.beta = beta

    def create_solution(self):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], velocity, mass]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        position = self.amend_position(position)
        fitness = self.get_fitness_position(position=position)
        velocity = np.random.uniform(self.problem.lb, self.problem.ub)
        mass = 0.0
        return [position, fitness, velocity, mass]

    def amend_position(self, position=None):
        """
        If solution out of bound at dimension x, then it will re-arrange to random location in the range of domain

        Args:
            position: vector position (location) of the solution.

        Returns:
            Amended position
        """
        return np.where(np.logical_and(self.problem.lb <= position, position <= self.problem.ub),
                        position, np.random.uniform(self.problem.lb, self.problem.ub))

    def _update_mass__(self, population):
        fit_total, fit_best, fit_worst = self.get_special_fitness(population)
        for agent in population:
            agent[self.ID_MAS] = np.exp((agent[self.ID_TAR][self.ID_FIT] - fit_best) / (fit_worst - fit_best + self.EPSILON)) / fit_total
        return population

    def _find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin((iteration + 1) / self.epoch * np.pi / 2)
        rsmax = 1.24
        if radius / average_dist < rsmin:
            rs = rsmin
        else:
            if radius / average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
        return potential

    def _acceleration__(self, population, g_best, iteration):
        eps = 2 ** (-52)
        pop = self._update_mass__(population)

        G = np.exp(-20.0 * (iteration + 1) / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * ((iteration + 1) / self.epoch) ** 0.5) + 1
        if self.problem.minmax == "min":
            k_best_pop = deepcopy(sorted(pop, key=lambda agent: agent[self.ID_MAS], reverse=True)[:k_best])
        else:
            k_best_pop = deepcopy(sorted(pop, key=lambda agent: agent[self.ID_MAS])[:k_best])
        mk_average = np.mean([item[self.ID_POS] for item in k_best_pop])

        acc_list = np.zeros((self.pop_size, self.problem.n_dims))
        for i in range(0, self.pop_size):
            dist_average = np.linalg.norm(pop[i][self.ID_POS] - mk_average)
            temp = np.zeros((self.problem.n_dims))

            for atom in k_best_pop:
                # calculate LJ-potential
                radius = np.linalg.norm(pop[i][self.ID_POS] - atom[self.ID_POS])
                potential = self._find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * np.random.uniform(0, 1, self.problem.n_dims) * ((atom[self.ID_POS] - pop[i][self.ID_POS]) / (radius + eps))
            temp = self.alpha * temp + self.beta * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            # calculate acceleration
            acc = G * temp / pop[i][self.ID_MAS]
            acc_list[i] = acc
        return acc_list

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate acceleration.
        atom_acc_list = self._acceleration__(self.pop, self.g_best, iteration=epoch)

        # Update velocity based on random dimensions and position of global best
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            velocity_rand = np.random.uniform(self.problem.lb, self.problem.ub)
            velocity = velocity_rand * self.pop[idx][self.ID_VEL] + atom_acc_list[idx]
            pos_new = self.pop[idx][self.ID_POS] + velocity
            # Relocate atom out of range
            agent[self.ID_POS] = self.amend_position(pos_new)
            pop_new.append(agent)
        pop_new = self.update_fitness_population(pop_new)
        pop_new = self.greedy_selection_population(self.pop, pop_new)

        _, current_best = self.get_global_best_solution(pop_new)
        if self.compare_agent(self.g_best, current_best):
            pop_new[np.random.randint(0, self.pop_size)] = deepcopy(self.g_best)
        self.pop = pop_new
