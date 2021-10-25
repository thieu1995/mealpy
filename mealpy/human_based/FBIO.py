#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:57, 14/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseFBIO(Optimizer):
    """
    My modified version of: Forensic-Based Investigation Optimization (FBIO)
        (FBI inspired meta-optimization)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S1568494620302799
    Notes:
        + Implement the fastest way (Remove all third loop)
        + Change equations
        + Change the flow of algorithm
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 4 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def probability(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = np.max(list_fitness)
        min1 = np.min(list_fitness)
        prob = (max1 - list_fitness) / (max1 - min1 + self.EPSILON)
        return prob

    def create_child1(self, idx, pop):
        n_change = np.random.randint(0, self.problem.n_dims)
        nb1, nb2 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
        # Eq.(2) in FBI Inspired Meta - Optimization
        pos_a = pop[idx][self.ID_POS].copy()
        pos_a[n_change] = pop[idx][self.ID_POS][n_change] + np.random.normal() * \
                          (pop[idx][self.ID_POS][n_change] -  (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change]) / 2)
        pos_a = self.amend_position_random(pos_a)
        fit_a = self.get_fitness_position(pos_a)
        if self.compare_agent([pos_a, fit_a], pop[idx]):
            return [pos_a, fit_a]
        return pop[idx].copy()

    def create_child2(self, idx, pop, g_best, prob):
        if np.random.uniform() > prob[idx]:
            r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
            pos_a = pop[idx][self.ID_POS].copy()
            temp = g_best[self.ID_POS] + pop[r1][self.ID_POS] + np.random.uniform() * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
            pos_a = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, temp, pos_a)
            pos_a = self.amend_position_random(pos_a)
            fit_a = self.get_fitness_position(pos_a)
            if self.compare_agent([pos_a, fit_a], pop[idx]):
                return [pos_a, fit_a]
        return pop[idx].copy()

    def create_child3(self, idx, pop, g_best):
        ### Remove third loop here also
        ### Eq.(6) in FBI Inspired Meta-Optimization
        pos_b = np.random.uniform(0, 1, self.problem.n_dims) * pop[idx][self.ID_POS] + \
                np.random.uniform(0, 1, self.problem.n_dims) * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_b = self.amend_position_random(pos_b)
        fit_b = self.get_fitness_position(pos_b)
        if self.compare_agent([pos_b, fit_b], pop[idx]):
            return [pos_b, fit_b]
        return pop[idx].copy()

    def create_child4(self, idx, pop, g_best):
        rr = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
        if self.compare_agent(pop[idx], pop[rr]):
            ## Eq.(7) in FBI Inspired Meta-Optimization
            pos_b = pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (pop[rr][self.ID_POS] - pop[idx][self.ID_POS]) + \
                    np.random.uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
        else:
            ## Eq.(8) in FBI Inspired Meta-Optimization
            pos_b = pop[idx][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (pop[idx][self.ID_POS] - pop[rr][self.ID_POS]) + \
                    np.random.uniform() * (g_best[self.ID_POS] - pop[idx][self.ID_POS])
        pos_b = self.amend_position_random(pos_b)
        fit_b = self.get_fitness_position(pos_b)
        if self.compare_agent([pos_b, fit_b], pop[idx]):
            return [pos_b, fit_b]
        return pop[idx].copy()

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        pop_copy = pop.copy()
        pop_idx = np.array(range(0, self.pop_size))

        if mode == "thread":
            # Investigation team - team A
            # Step A1
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child1, pop=pop_copy), pop_idx)
            pop_new = [x for x in pop_child]
            list_fitness = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop_new])
            prob = self.probability(list_fitness)

            # Step A2
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop_new, g_best=g_best, prob=prob), pop_idx)
            pop_new = [x for x in pop_child]

            ## Persuing team - team B
            ## Step B1
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child3, pop=pop_new, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]

            ## Step B2
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child4, pop=pop_new, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]
        elif mode == "process":
            # Investigation team - team A
            # Step A1
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child1, pop=pop_copy), pop_idx)
            list_fitness = np.array([item[self.ID_FIT] for item in pop_child])
            prob = self.probability(list_fitness)
            # Step A2
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=pop_child, g_best=g_best, prob=prob), pop_idx)
            pop_new = [x for x in pop_child]

            ## Persuing team - team B
            ## Step B1
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child3, pop=pop_new, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]
            ## Step B2
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child4, pop=pop_new, g_best=g_best), pop_idx)
            pop_new = [x for x in pop_child]
        else:
            # Investigation team - team A
            # Step A1
            pop_child = [self.create_child1(idx, pop_copy) for idx in pop_idx]
            list_fitness = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop_child])
            prob = self.probability(list_fitness)
            # Step A2
            pop_child = [self.create_child2(idx, pop_child, g_best, prob) for idx in pop_idx]

            ## Persuing team - team B
            ## Step B1
            pop_child = [self.create_child3(idx, pop_child, g_best) for idx in pop_idx]
            ## Step B2
            pop_new = [self.create_child4(idx, pop_child, g_best) for idx in pop_idx]
        return pop_new


class OriginalFBIO(BaseFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)
        (FBI inspired meta-optimization)
    Link:
        DOI: https://doi.org/10.1016/j.asoc.2020.106339
        Matlab code: https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (Harmony Memory Size), default = 100
        """
        super().__init__(problem, **kwargs)
        self.nfe_per_epoch = 4 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        if mode != "sequential":
            print("FBIO algorithm is support sequential mode only!")
            exit(0)

        # Investigation team - team A
        # Step A1
        for i in range(0, self.pop_size):
            n_change = np.random.randint(0, self.problem.n_dims)
            nb1, nb2 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
            # Eq.(2) in FBI Inspired Meta - Optimization
            pos_a = pop[i][self.ID_POS].copy()
            pos_a[n_change] = pop[i][self.ID_POS][n_change] + (np.random.uniform() - 0.5) * 2 * (pop[i][self.ID_POS][n_change] -
                                            (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change]) / 2)
            ## Not good move here, change only 1 variable but check bound of all variable in solution
            pos_a = self.amend_position_random(pos_a)
            fit_a = self.get_fitness_position(pos_a)
            if self.compare_agent([pos_a, fit_a], pop[i]):
                pop[i] = [pos_a, fit_a]
        # Step A2
        list_fitness = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop])
        prob = self.probability(list_fitness)
        for i in range(0, self.pop_size):
            if np.random.uniform() > prob[i]:
                r1, r2, r3 = np.random.choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                pos_a = pop[i][self.ID_POS].copy()
                Rnd = np.floor(np.random.uniform() * self.problem.n_dims) + 1

                for j in range(0, self.problem.n_dims):
                    if (np.random.uniform() < np.random.uniform() or Rnd == j):
                        pos_a[j] = g_best[self.ID_POS][j] + pop[r1][self.ID_POS][j] + np.random.uniform() * (pop[r2][self.ID_POS][j] - pop[r3][self.ID_POS][j])
                    ## In the original matlab code they do the else condition here, not good again because no need else here
                ## Same here, they do check the bound of all variable in solution
                pos_a = self.amend_position_random(pos_a)
                fit_a = self.get_fitness_position(pos_a)
                if self.compare_agent([pos_a, fit_a], pop[i]):
                    pop[i] = [pos_a, fit_a]
        ## Persuing team - team B
        ## Step B1
        for i in range(0, self.pop_size):
            pos_b = pop[i][self.ID_POS].copy()
            for j in range(0, self.problem.n_dims):
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b[j] = np.random.uniform() * pop[i][self.ID_POS][j] + np.random.uniform() * (g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
            pos_b = self.amend_position_random(pos_b)
            fit_b = self.get_fitness_position(pos_b)
            if self.compare_agent([pos_b, fit_b], pop[i]):
                pop[i] = [pos_b, fit_b]

        ## Step B2
        for i in range(0, self.pop_size):
            ### Not good move here again
            rr = np.random.randint(0, self.pop_size)
            while rr == i:
                rr = np.random.randint(0, self.pop_size)
            if self.compare_agent(pop[i], pop[rr]):
                ## Eq.(7) in FBI Inspired Meta-Optimization
                pos_b = pop[i][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (pop[rr][self.ID_POS] - pop[i][self.ID_POS]) + \
                        np.random.uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
            else:
                ## Eq.(8) in FBI Inspired Meta-Optimization
                pos_b = pop[i][self.ID_POS] + np.random.uniform(0, 1, self.problem.n_dims) * (pop[i][self.ID_POS] - pop[rr][self.ID_POS]) + \
                        np.random.uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
            pos_b = self.amend_position_random(pos_b)
            fit_b = self.get_fitness_position(pos_b)
            if self.compare_agent([pos_b, fit_b], pop[i]):
                pop[i] = [pos_b, fit_b]
        return pop