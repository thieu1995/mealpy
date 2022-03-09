# !/usr/bin/env python
# Created by "Thieu" at 07:44, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class ImprovedBSO(Optimizer):
    """
    My improved version of: Brain Storm Optimization (BSO)

    Notes
    ~~~~~
    + Remove some probability parameters, and some useless equations.
    + Add Levy-flight technique for more robust

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): 25% percent
        + p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
        + p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
        + p4 (float): [0.4, 0.6], Need more weights on the centers instead of the random position

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.BSO import ImprovedBSO
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
    >>> m_clusters = 5
    >>> p1 = 0.25
    >>> p2 = 0.5
    >>> p3 = 0.75
    >>> p4 = 0.6
    >>> model = ImprovedBSO(problem_dict1, epoch, pop_size, m_clusters, p1, p2, p3, p4)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): 25% percent
            p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
            p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
            p4 (float): Need more weights on the centers instead of the random position
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.m_clusters = m_clusters
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.m_solution = int(self.pop_size / self.m_clusters)

        self.pop_group, self.centers = None, None

    def _find_cluster(self, pop_group):
        centers = []
        for i in range(0, self.m_clusters):
            _, local_best = self.get_global_best_solution(pop_group[i])
            centers.append(deepcopy(local_best))
        return centers

    def _make_group(self, pop):
        pop_group = []
        for idx in range(0, self.m_clusters):
            pop_group.append(deepcopy(pop[idx * self.m_solution:(idx + 1) * self.m_solution]))
        return pop_group

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.pop_group = self._make_group(self.pop)
        self.centers = self._find_cluster(self.pop_group)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        epxilon = 1 - 1 * (epoch + 1) / self.epoch  # 1. Changed here, no need: k

        if np.random.uniform() < self.p1:  # p_5a
            idx = np.random.randint(0, self.m_clusters)
            solution_new = self.create_solution()
            self.centers[idx] = solution_new

        pop_group = deepcopy(self.pop_group)
        for i in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(i / self.m_solution)
            location_id = int(i % self.m_solution)

            if np.random.uniform() < self.p2:  # p_6b
                if np.random.uniform() < self.p3:
                    pos_new = self.centers[cluster_id][self.ID_POS] + epxilon * np.random.uniform()
                else:  # 2. Using levy flight here
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, case=-1)
                    pos_new = self.pop_group[cluster_id][location_id][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * levy_step
            else:
                id1, id2 = np.random.choice(range(0, self.m_clusters), 2, replace=False)
                if np.random.uniform() < self.p4:
                    pos_new = 0.5 * (self.centers[id1][self.ID_POS] + self.centers[id2][self.ID_POS]) + epxilon * np.random.uniform()
                else:
                    rand_id1 = np.random.randint(0, self.m_solution)
                    rand_id2 = np.random.randint(0, self.m_solution)
                    pos_new = 0.5 * (self.pop_group[id1][rand_id1][self.ID_POS] + self.pop_group[id2][rand_id2][self.ID_POS]) + \
                              epxilon * np.random.uniform()
            pos_new = self.amend_position(pos_new)
            pop_group[cluster_id][location_id] = [pos_new, None]
        pop_group = [self.update_fitness_population(group) for group in pop_group]
        for idx in range(0, self.m_clusters):
            self.pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx])

        # Needed to update the centers and population
        self.centers = self._find_cluster(self.pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += self.pop_group[idx]


class BaseBSO(ImprovedBSO):
    """
    The original version of: Brain Storm Optimization (BSO)

    Links:
        1. https://doi.org/10.1007/978-3-642-21515-5_36

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): [0.1, 0.5], probability
        + p2 (float): [0.5, 0.95], probability
        + p3 (float): [0.2, 0.8], probability
        + p4 (float): [0.2, 0.8], probability
        + slope (int): [10, 15, 20, 25], changing logsig() function's slope (k: in the paper)
        + miu (float): [0], mean of normal distribution (gaussian)
        + xichma (float): [1], standard deviation of normal distribution (gaussian)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.BSO import BaseBSO
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
    >>> m_clusters = 5
    >>> p1 = 0.2
    >>> p2 = 0.8
    >>> p3 = 0.4
    >>> p4 = 0.5
    >>> slope = 20
    >>> miu = 0
    >>> xichma = 1
    >>> model = BaseBSO(problem_dict1, epoch, pop_size, m_clusters, p1, p2, p3, p4, slope, miu, xichma)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Y., 2011, June. Brain storm optimization algorithm. In International
    conference in swarm intelligence (pp. 303-309). Springer, Berlin, Heidelberg.
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 m_clusters=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, slope=20, miu=0, xichma=1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): probability
            p2 (float): probability
            p3 (float): probability
            p4 (float): probability
            slope (int): changing logsig() function's slope (k: in the paper)
            miu (float): mean of normal distribution (gaussian)
            xichma (float): standard deviation of normal distribution (gaussian)
        """
        super().__init__(problem, epoch, pop_size, m_clusters, p1, p2, p3, p4, **kwargs)
        self.slope = slope
        self.miu = miu
        self.xichma = xichma

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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        x = (0.5 * self.epoch - (epoch + 1)) / self.slope
        epxilon = np.random.uniform() * (1 / (1 + np.exp(-x)))

        if np.random.rand() < self.p1:  # p_5a
            idx = np.random.randint(0, self.m_clusters)
            solution_new = self.create_solution()
            self.centers[idx] = solution_new

        pop_group = deepcopy(self.pop_group)
        for i in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(i / self.m_solution)
            location_id = int(i % self.m_solution)

            if np.random.uniform() < self.p2:  # p_6b
                if np.random.uniform() < self.p3:  # p_6i
                    cluster_id = np.random.randint(0, self.m_clusters)
                if np.random.uniform() < self.p3:
                    pos_new = self.centers[cluster_id][self.ID_POS] + epxilon * np.random.normal(self.miu, self.xichma)
                else:
                    rand_idx = np.random.randint(0, self.m_solution)
                    pos_new = self.pop_group[cluster_id][rand_idx][self.ID_POS] + np.random.uniform()
            else:
                id1, id2 = np.random.choice(range(0, self.m_clusters), 2, replace=False)
                if np.random.uniform() < self.p4:
                    pos_new = 0.5 * (self.centers[id1][self.ID_POS] + self.centers[id2][self.ID_POS]) + \
                              epxilon * np.random.normal(self.miu, self.xichma)
                else:
                    rand_id1 = np.random.randint(0, self.m_solution)
                    rand_id2 = np.random.randint(0, self.m_solution)
                    pos_new = 0.5 * (self.pop_group[id1][rand_id1][self.ID_POS] + self.pop_group[id2][rand_id2][self.ID_POS]) + \
                              epxilon * np.random.normal(self.miu, self.xichma)
            pos_new = self.amend_position(pos_new)
            pop_group[cluster_id][location_id] = [pos_new, None]
        pop_group = [self.update_fitness_population(group) for group in pop_group]
        for idx in range(0, self.m_clusters):
            self.pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx])

        # Needed to update the centers and population
        self.centers = self._find_cluster(self.pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += self.pop_group[idx]

