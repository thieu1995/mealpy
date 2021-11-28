#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 16:58, 08/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseGSKA(Optimizer):
    """
    My version of: Gaining Sharing Knowledge-based Algorithm (GSKA)
        (Gaining‑sharing Knowledge-Based Algorithm For Solving Optimization Problems: A Novel Nature‑inspired Algorithm)
    Link:
        DOI: https://doi.org/10.1007/s13042-019-01053-x
    Notes:
        + Remove all third loop
        + Solution represent junior or senior instead of dimension of solution
        + Remove 2 parameters
        + Change some equations for large-scale optimization
        + Apply the ideas of levy-flight and global best
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.1, kr=0.7, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kr (float): knowledge ratio, default = 0.7
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pb = pb
        self.kr = kr

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        D = int(np.ceil(self.pop_size * (1 - (epoch + 1) / self.epoch)))
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1

            if idx < D:  # senior gaining and sharing
                if np.random.uniform() <= self.kr:
                    rand_idx = np.random.choice(list(set(range(0, self.pop_size)) - {previ, idx, nexti}))
                    if self.compare_agent(self.pop[rand_idx], self.pop[idx]):
                        pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[previ][self.ID_POS] - self.pop[nexti][self.ID_POS] +
                                   self.pop[rand_idx][self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        pos_new = self.g_best[self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_idx][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            else:  # junior gaining and sharing
                if np.random.uniform() <= self.kr:
                    id1 = int(self.pb * self.pop_size)
                    id2 = id1 + int(self.pop_size - 2 * 100 * self.pb)
                    rand_best = np.random.choice(list(set(range(0, id1)) - {idx}))
                    rand_worst = np.random.choice(list(set(range(id2, self.pop_size)) - {idx}))
                    rand_mid = np.random.choice(list(set(range(id1, id2)) - {idx}))
                    if self.compare_agent(self.pop[rand_mid], self.pop[idx]):
                        pos_new = self.pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_best][self.ID_POS] - self.pop[rand_worst][self.ID_POS] +
                                   self.pop[rand_mid][self.ID_POS] - self.pop[idx][self.ID_POS])
                    else:
                        pos_new = self.g_best[self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * \
                                  (self.pop[rand_mid][self.ID_POS] - self.pop[idx][self.ID_POS])
                else:
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)


class OriginalGSKA(Optimizer):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)
        (Gaining‑sharing Knowledge-Based Algorithm For Solving Optimization Problems: A Novel Nature‑inspired Algorithm)
    Link:
        DOI: https://doi.org/10.1007/s13042-019-01053-x
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pb=0.1, kf=0.5, kr=0.9, k=5, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kf (float): knowledge factor that controls the total amount of gained and shared knowledge added
                        from others to the current individual during generations, default = 0.5
            kr (float): knowledge ratio, default = 0.9
            k (int): Number of generations effect to D-dimension, default = 5
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pb = pb
        self.kf = kf
        self.kr = kr
        self.k = k

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        D = int(self.problem.n_dims * (1 - (epoch + 1) / self.epoch) ** self.k)
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1

            # The random individual is for all dimension values
            rand_idx = np.random.choice(list(set(range(0, self.pop_size)) - {previ, idx, nexti}))
            pos_new = deepcopy(self.pop[idx][self.ID_POS])

            for j in range(0, self.problem.n_dims):
                if j < D:  # junior gaining and sharing
                    if np.random.uniform() <= self.kr:
                        if self.compare_agent(self.pop[rand_idx], self.pop[idx]):
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[previ][self.ID_POS][j] - self.pop[nexti][self.ID_POS][j] +
                                          self.pop[rand_idx][self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
                        else:
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[previ][self.ID_POS][j] - self.pop[nexti][self.ID_POS][j] +
                                          self.pop[idx][self.ID_POS][j] - self.pop[rand_idx][self.ID_POS][j])
                else:  # senior gaining and sharing
                    if np.random.uniform() <= self.kr:
                        id1 = int(self.pb * self.pop_size)
                        id2 = id1 + int(self.pop_size - 2 * 100 * self.pb)
                        rand_best = np.random.choice(list(set(range(0, id1)) - {idx}))
                        rand_worst = np.random.choice(list(set(range(id2, self.pop_size)) - {idx}))
                        rand_mid = np.random.choice(list(set(range(id1, id2)) - {idx}))
                        if self.compare_agent(self.pop[rand_mid], self.pop[idx]):
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[rand_best][self.ID_POS][j] - self.pop[rand_worst][self.ID_POS][j] +
                                          self.pop[rand_mid][self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
                        else:
                            pos_new[j] = self.pop[idx][self.ID_POS][j] + self.kf * \
                                         (self.pop[rand_best][self.ID_POS][j] - self.pop[rand_worst][self.ID_POS][j] +
                                          self.pop[idx][self.ID_POS][j] - self.pop[rand_mid][self.ID_POS][j])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
