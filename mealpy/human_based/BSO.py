#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:44, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, choice, normal
from math import exp
from copy import deepcopy
from mealpy.optimizer import Root


class BaseBSO(Root):
    """
    The original version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Link:
        DOI: https://doi.org/10.1007/978-3-642-21515-5_36
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 m=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, k=20, miu=0, xichma=1, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size  # n: pop_size,
        self.m = m  #  m: clusters
        self.p1 = p1  # probability
        self.p2 = p2  # probability
        self.p3 = p3  # probability
        self.p4 = p4  # probability
        self.k = k  # changing logsig() function's slope
        self.miu = miu
        self.xichma = xichma
        self.m_solution = int(self.pop_size / self.m)

    def _creat_population__(self):
        pop = []
        for i in range(0, self.m):
            group = [self.create_solution() for _ in range(0, self.m_solution)]
            pop.append(group)
        return pop

    def _find_cluster__(self, pop):
        centers = []
        for i in range(0, self.m):
            pop_sorted = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            centers.append(deepcopy(pop_sorted[self.ID_MIN_PROB]))
        return centers

    def train(self):
        pop = self._creat_population__()
        centers = self._find_cluster__(pop)
        g_best = self.get_global_best_solution(centers, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            x = (0.5 * self.epoch - (epoch+1)) / self.k
            epxilon = uniform() * (1 / (1 + exp(-x)))

            if uniform() < self.p1:  # p_5a
                idx = randint(0, self.m)
                solution_new = self.create_solution()
                centers[idx] = solution_new

            for i in range(0, self.pop_size):  # Generate new individuals
                cluster_id = int(i / self.m_solution)
                location_id = int(i % self.m_solution)

                if uniform() < self.p2:  # p_6b
                    if uniform() < self.p3:  # p_6i
                        cluster_id = randint(0, self.m)
                    if uniform() < self.p3:
                        pos_new = centers[cluster_id][self.ID_POS] + epxilon * normal(self.miu, self.xichma)
                    else:
                        rand_idx = randint(0, self.m_solution)
                        pos_new = pop[cluster_id][rand_idx][self.ID_POS] + uniform()
                else:
                    id1, id2 = choice(range(0, self.m), 2, replace=False)
                    if uniform() < self.p4:
                        pos_new = 0.5 * (centers[id1][self.ID_POS] + centers[id2][self.ID_POS]) + epxilon * normal(self.miu, self.xichma)
                    else:
                        rand_id1 = randint(0, self.m_solution)
                        rand_id2 = randint(0, self.m_solution)
                        pos_new = 0.5 * (pop[id1][rand_id1][self.ID_POS] + pop[id2][rand_id2][self.ID_POS]) + epxilon * normal(self.miu, self.xichma)
                pos_new = self.amend_position_random(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[cluster_id][location_id][self.ID_FIT]:
                    pop[cluster_id][location_id] = [pos_new, fit]

            # Needed to update the centers and global best
            centers = self._find_cluster__(pop)
            g_best = self.update_global_best_solution(centers, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedBSO(Root):
    """
    My improved version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Notes:
        + No need some parameters, and some useless equations
        + Using levy-flight for more robust
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 m=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.m = m              # n: pop_size, m: clusters
        self.p1 = p1            # 25% percent
        self.p2 = p2            # 50% percent changed by its own (local search), 50% percent changed by outside (global search)
        self.p3 = p3            # 75% percent develop the old idea, 25% invented new idea based on levy-flight
        self.p4 = p4            # Need more weights on the centers instead of the random position
        self.m_solution = int(self.pop_size / self.m)

    def _creat_population__(self):
        pop = []
        for i in range(0, self.m):
            group = [self.create_solution() for _ in range(0, self.m_solution)]
            pop.append(group)
        return pop

    def _find_cluster__(self, pop):
        centers = []
        for i in range(0, self.m):
            pop_sorted = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            centers.append(deepcopy(pop_sorted[self.ID_MIN_PROB]))
        return centers


    def train(self):
        pop = self._creat_population__()
        centers = self._find_cluster__(pop)
        g_best = self.get_global_best_solution(centers, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            epxilon = 1 - 1 * (epoch + 1) / self.epoch                          # 1. Changed here, no need: k

            if uniform() < self.p1:                                             # p_5a
                idx = randint(0, self.m)
                solution_new = self.create_solution()
                centers[idx] = solution_new

            for i in range(0, self.pop_size):                                   # Generate new individuals
                cluster_id = int(i / self.m_solution)
                location_id = int(i % self.m_solution)

                if uniform() < self.p2:                                         # p_6b
                    if uniform() < self.p3:
                        pos_new = centers[cluster_id][self.ID_POS] + epxilon * uniform()
                    else:                                                       # 2. Using levy flight here
                        pos_new = self.levy_flight(epoch, pop[cluster_id][location_id][self.ID_POS], g_best[self.ID_POS])
                else:
                    id1, id2 = choice(range(0, self.m), 2, replace=False)
                    if uniform() < self.p4:
                        pos_new = 0.5 * (centers[id1][self.ID_POS] + centers[id2][self.ID_POS]) + epxilon * uniform()
                    else:
                        rand_id1 = randint(0, self.m_solution)
                        rand_id2 = randint(0, self.m_solution)
                        pos_new = 0.5 * (pop[id1][rand_id1][self.ID_POS] + pop[id2][rand_id2][self.ID_POS]) + epxilon * uniform()
                pos_new = self.amend_position_random(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[cluster_id][location_id][self.ID_FIT]:
                    pop[cluster_id][location_id] = [pos_new, fit]

            # Needed to update the centers and global best
            centers = self._find_cluster__(pop)
            g_best = self.update_global_best_solution(centers, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
