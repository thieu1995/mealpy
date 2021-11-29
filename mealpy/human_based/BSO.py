#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:44, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class ImprovedBSO(Optimizer):
    """
    My improved version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Notes:
        + No need some parameters, and some useless equations
        + Using levy-flight for more robust
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 m_clusters=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (): number of clusters (m in the paper)
            p1 (): 25% percent
            p2 (): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
            p3 (): 75% percent develop the old idea, 25% invented new idea based on levy-flight
            p4 (): Need more weights on the centers instead of the random position
            **kwargs ():
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
            pos_new = self.amend_position_random(pos_new)
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
        (Brain storm optimization algorithm)
    Link:
        DOI: https://doi.org/10.1007/978-3-642-21515-5_36
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 m_clusters=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, slope=20, miu=0, xichma=1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): probability
            p2 (float): probability
            p3 (float): probability
            p4 (float): probability
            slope (int): changing logsig() function's slope (k: in the paper)
            miu (float):
            xichma (float):
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, m_clusters, p1, p2, p3, p4, **kwargs)
        self.slope = slope
        self.miu = miu
        self.xichma = xichma

    def evolve(self, epoch):
        """
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
            pos_new = self.amend_position_random(pos_new)
            pop_group[cluster_id][location_id] = [pos_new, None]
        pop_group = [self.update_fitness_population(group) for group in pop_group]
        for idx in range(0, self.m_clusters):
            self.pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx])

        # Needed to update the centers and population
        self.centers = self._find_cluster(self.pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += self.pop_group[idx]

