#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:44, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, choice, normal
import numpy as np
from math import exp
from copy import deepcopy
from mealpy.root import Root
from sklearn.cluster import KMeans


class BaseBSO(Root):
    """
    The original version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Link:
        DOI: https://doi.org/10.1007/978-3-642-21515-5_36
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100,
                 m=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, k=20, miu=0, xichma=1):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size  # n: pop_size, m: clusters
        self.m = m
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.k = k
        self.miu = miu
        self.xichma = xichma
        self.m_solution = int(self.pop_size / self.m)

    def _creat_population__(self):
        pop = []
        for i in range(0, self.m):
            group = [self._create_solution__() for _ in range(0, self.m_solution)]
            pop.append(group)
        return pop

    def _find_cluster__(self, pop):
        centers = []
        for i in range(0, self.m):
            pop_sorted = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            centers.append(deepcopy(pop_sorted[self.ID_MIN_PROB]))
        return centers

    def _train__(self):
        pop = self._creat_population__()
        centers = self._find_cluster__(pop)
        g_best = self._get_global_best__(centers, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            x = (0.5 * self.epoch - (epoch+1)) / self.k
            epxilon = uniform() * (1 / (1 + exp(-x)))

            if uniform() < self.p1:  # p_5a
                idx = randint(0, self.m)
                solution_new = self._create_solution__()
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
                pos_new = self._amend_solution_random_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[cluster_id][location_id][self.ID_FIT]:
                    pop[cluster_id][location_id] = [pos_new, fit]

            # Needed to update the centers and global best
            centers = self._find_cluster__(pop)
            g_best = self._update_global_best__(centers, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class ImprovedBSO(Root):
    """
    My improved version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Noted:
        + No need some parameters, and some useless equations
        + Using levy-flight for more robust
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, m=5, p1=0.25, p2=0.5, p3=0.75, p4=0.5):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
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
            group = [self._create_solution__() for _ in range(0, self.m_solution)]
            pop.append(group)
        return pop

    def _find_cluster__(self, pop):
        centers = []
        for i in range(0, self.m):
            pop_sorted = sorted(pop[i], key=lambda item: item[self.ID_FIT])
            centers.append(deepcopy(pop_sorted[self.ID_MIN_PROB]))
        return centers

    def _train__(self):
        pop = self._creat_population__()
        centers = self._find_cluster__(pop)
        g_best = self._get_global_best__(centers, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            epxilon = 1 - 1 * (epoch + 1) / self.epoch                          # 1. Changed here, no need: k

            if uniform() < self.p1:                                             # p_5a
                idx = randint(0, self.m)
                solution_new = self._create_solution__()
                centers[idx] = solution_new

            for i in range(0, self.pop_size):                                   # Generate new individuals
                cluster_id = int(i / self.m_solution)
                location_id = int(i % self.m_solution)

                if uniform() < self.p2:                                         # p_6b
                    if uniform() < self.p3:
                        pos_new = centers[cluster_id][self.ID_POS] + epxilon * uniform()
                    else:                                                       # 2. Using levy flight here
                        pos_new = self._levy_flight__(epoch, pop[cluster_id][location_id][self.ID_POS], g_best[self.ID_POS])
                else:
                    id1, id2 = choice(range(0, self.m), 2, replace=False)
                    if uniform() < self.p4:
                        pos_new = 0.5 * (centers[id1][self.ID_POS] + centers[id2][self.ID_POS]) + epxilon * uniform()
                    else:
                        rand_id1 = randint(0, self.m_solution)
                        rand_id2 = randint(0, self.m_solution)
                        pos_new = 0.5 * (pop[id1][rand_id1][self.ID_POS] + pop[id2][rand_id2][self.ID_POS]) + epxilon * uniform()
                pos_new = self._amend_solution_random_faster__(pos_new)
                fit = self._fitness_model__(pos_new)
                if fit < pop[cluster_id][location_id][self.ID_FIT]:
                    pop[cluster_id][location_id] = [pos_new, fit]

            # Needed to update the centers and global best
            centers = self._find_cluster__(pop)
            g_best = self._update_global_best__(centers, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

    
class KmeansBSO(Root):
    """
    The original version of: Brain Storm Optimization (BSO)
        (Brain storm optimization algorithm)
    Link:
        DOI: https://doi.org/10.1007/978-3-642-21515-5_36
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100,
                 m=5, p1=0.2, p2=0.8, p3=0.4, p4=0.5, k=20, miu=0, xichma=1):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size  # n: pop_size, m: clusters
        self.m = m
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.k = k
        self.miu = miu
        self.xichma = xichma
        self.m_solution = int(self.pop_size / self.m)

    def _create_population__(self):
        pop = []
        for i in range(0, self.m):
            group = [self._create_solution__() for _ in range(0, self.m_solution)]
            pop.append(group)
        return pop

    def _find_cluster__(self, pop):
        pop_km = np.concatenate(pop)
        kmeans = KMeans(n_clusters=self.m, random_state=42).fit(np.vstack(pop_km[:, 0]))
        clusters = kmeans.predict(np.vstack(pop_km[:, 0]))

        pop = []
        for i in range(self.m):
            pop.append(pop_km[clusters == i])

        centers = [[center, self._fitness_model__(solution=center, minmax=0)] for center in kmeans.cluster_centers_]

        return pop, centers, clusters

    def _train__(self):
        pop = self._create_population__() # Step 1
        pop, centers, clusters = self._find_cluster__(pop) # Step 2
        g_best = self._get_global_best__(centers, self.ID_FIT, self.ID_MIN_PROB) # Step 3,4

        for epoch in range(self.epoch):
            x = (0.5 * self.epoch - (epoch + 1)) / self.k
            epxilon = uniform() * (1 / (1 + exp(-x)))

            if uniform() < self.p1:  # p_5a # Step 5a
                idx = randint(0, self.m) # Step 5 a i
                solution_new = self._create_solution__() # Step 5 a ii
                centers[idx] = solution_new

            for i in range(0, self.pop_size):  # Generate new individuals # Step 6
                cluster_id = int(clusters[i])

                location_id = len(clusters[:i][clusters[:i] == cluster_id])

                if uniform() < self.p2:  # p_6b # Step 6 b
                    store_cluster = cluster_id                    
                    cluster_id = choice(range(0, self.m), 1, p=[len(p)/self.pop_size for p in pop])[0] # Step 6 b i

                    if uniform() < self.p3: # Step 6 b iii 

                        pos_new = centers[cluster_id][self.ID_POS] + epxilon * normal(self.miu, self.xichma) # Step 6 b iii 1

                    else:
                        rand_idx = randint(0, len(clusters[clusters == cluster_id]))
                        pos_new = pop[cluster_id][rand_idx][self.ID_POS] + uniform()  # Step 6 b iv
                    cluster_id = store_cluster
                else: # Step 6 c 
                    id1, id2 = choice(range(0, self.m), 2, replace=False)
                    if uniform() < self.p4: # Step 6 c i
                        pos_new = 0.5 * (centers[id1][self.ID_POS] + centers[id2][self.ID_POS]) + epxilon * normal(
                            self.miu, self.xichma)  # Step 6 c ii
                    else:  # Step 6 c iii
                        rand_id1 = randint(0, len(clusters[clusters == id1]))
                        rand_id2 = randint(0, len(clusters[clusters == id2]))

                        pos_new = 0.5 * (pop[id1][rand_id1][self.ID_POS] + pop[id2][rand_id2][
                            self.ID_POS]) + epxilon * normal(self.miu, self.xichma)
                pos_new = self._amend_solution_random_faster__(pos_new)
                fit = self._fitness_model__(pos_new)

                if fit < pop[cluster_id][location_id][self.ID_FIT]:
                    pop[cluster_id][location_id] = [pos_new, fit]

            # Needed to update the centers and global best
            pop, centers, clusters = self._find_cluster__(pop)
            g_best = self._update_global_best__(centers, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train