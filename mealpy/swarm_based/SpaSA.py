#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:22, 29/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseSpaSA(Optimizer):
    """
        My version of: Sparrow Search Algorithm (SpaSA)
            (A novel swarm intelligence optimization approach: sparrow search algorithm)
        Link:
            https://doi.org/10.1080/21642583.2019.1708830
        Noted:
            + First, I sort the algorithm and find g-best and g-worst
            + In Eq. 4, Instead of using A+ and L, I used np.random.normal().
            + Their algorithm 1 flow is missing all important component such as g_best_position, fitness updated,
            + After change some equations and flows --> this become the BEST algorithm
    """
    def __init__(self, problem, epoch=10000, pop_size=100, ST=0.8, PD=0.2, SD=0.1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value
            PD (float): number of producers (percentage)
            SD (float): number of sparrows who perceive the danger
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.ST = ST
        self.PD = PD
        self.SD = SD

        self.n1 = int(self.PD * self.pop_size)
        self.n2 = int(self.SD * self.pop_size)

        self.nfe_per_epoch = self.pop_size + self.n2
        self.sort_flag = True

    def create_child(self, idx, pop, r2):
        # Using equation (3) update the sparrow’s location;
        if idx < self.n1:
            if r2 < self.ST:
                x_new = pop[idx][self.ID_POS] * np.exp((idx + 1) / ((np.random.uniform() + self.EPSILON) * self.epoch))
            else:
                x_new = pop[idx][self.ID_POS] + np.random.normal() * np.ones(self.problem.n_dims)
        else:
        # Using equation (4) update the sparrow’s location;
            _, x_p, worst = self.get_special_solutions(pop, best=1, worst=1)
            g_best = x_p[0], g_worst = worst[0]
            if idx > int(self.pop_size / 2):
                x_new = np.random.normal() * np.exp((g_worst[self.ID_POS] - pop[idx][self.ID_POS]) / (idx + 1) ** 2)
            else:
                x_new = g_best[self.ID_POS] + np.abs(pop[idx][self.ID_POS] - g_best[self.ID_POS]) * np.random.normal()

        pos_new = self.amend_position_random(x_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def create_child2(self, idx, pop, g_best, g_worst):
        #  Using equation (5) update the sparrow’s location;
        if self.compare_agent(pop[idx], g_best):
            x_new = pop[idx][self.ID_POS] + np.random.uniform(-1, 1) * (np.abs(pop[idx][self.ID_POS] - g_worst[self.ID_POS]) /
                                (pop[idx][self.ID_FIT][self.ID_TAR] - g_worst[self.ID_FIT][self.ID_TAR] + self.EPSILON))
        else:
            x_new = g_best[self.ID_POS] + np.random.normal() * np.abs(pop[idx][self.ID_POS] - g_best[self.ID_POS])
        pos_new = self.amend_position_random(x_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
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
        r2 = np.random.uniform()  # R2 in [0, 1], the alarm value, random value
        pop_idx = np.array(range(0, self.pop_size))
        pop_idx2 = np.array(range(0, self.n2))

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, r2=r2), pop_idx)
            child = [x for x in pop_child]
            child, g_best, g_worst = self.get_special_solutions(child, best=1, worst=1)

            pop2 = child[self.n2:]
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=child, g_best=g_best[0], g_worst=g_worst[0]), pop_idx2)
            child2 = [x for x in pop_child]

        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, r2=r2), pop_idx)
            child = [x for x in pop_child]
            child, g_best, g_worst = self.get_special_solutions(child, best=1, worst=1)

            pop2 = child[self.n2:]
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child2, pop=child, g_best=g_best[0], g_worst=g_worst[0]), pop_idx2)
            child2 = [x for x in pop_child]

        else:
            child = [self.create_child(idx, pop, r2) for idx in pop_idx]
            child, g_best, g_worst = self.get_special_solutions(child, best=1, worst=1)
            pop2 = child[self.n2:]
            child2 = [self.create_child2(idx, child, g_best[0], g_worst[0]) for idx in pop_idx2]
        return child2 + pop2


class OriginalSpaSA(BaseSpaSA):
    """
        The original version of: Sparrow Search Algorithm
            (A novel swarm intelligence optimization approach: sparrow search algorithm)
        Link:
            https://doi.org/10.1080/21642583.2019.1708830
        Note:
            + Very weak algorithm
    """

    def __init__(self, problem, epoch=10000, pop_size=100, ST=0.8, PD=0.2, SD=0.1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value
            PD (float): number of producers (percentage)
            SD (float): number of sparrows who perceive the danger
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, ST, PD, SD, **kwargs)

    def create_child(self, idx, pop, r2):
        # Using equation (3) update the sparrow’s location;
        if idx < self.n1:
            if r2 < self.ST:
                x_new = pop[idx][self.ID_POS] * np.exp((idx + 1) / ((np.random.uniform() + self.EPSILON) * self.epoch))
            else:
                x_new = pop[idx][self.ID_POS] + np.random.normal() * np.ones(self.problem.n_dims)
        else:
            # Using equation (4) update the sparrow’s location;
            _, x_p, worst = self.get_special_solutions(pop, best=1, worst=1)
            g_best, g_worst = x_p[0], worst[0]
            if idx > int(self.pop_size / 2):
                x_new = np.random.normal() * np.exp((g_worst[self.ID_POS] - pop[idx][self.ID_POS]) / (idx + 1) ** 2)
            else:
                L = np.ones((1, self.problem.n_dims))
                A = np.sign(np.random.uniform(-1, 1, (1, self.problem.n_dims)))
                A1 = A.T * np.linalg.inv(np.matmul(A, A.T)) * L
                x_new = g_best[self.ID_POS] + np.matmul(np.abs(pop[idx][self.ID_POS] - g_best[self.ID_POS]), A1)
        pos_new = self.amend_position_random(x_new)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

