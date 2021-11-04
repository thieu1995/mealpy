#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseEOA(Optimizer):
    """
    My modified version of: Earthworm Optimisation Algorithm (EOA)
        (Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm for global optimisation problems)
    Link:
        http://doi.org/10.1504/IJBIC.2015.10004283
        https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa
    Notes:
        + The original version from matlab code above will not working well, even with small dimensions.
        + I changed updating process
        + Changed cauchy process using x_mean
        + Used global best solution
        + Remove third loop for faster
    """

    def __init__(self, problem, epoch=10000, pop_size=100, p_c=0.9, p_m=0.01, n_best=2, alpha=0.98, beta=1, gamma=0.9, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_c (): default = 0.9, crossover probability
            p_m (): default = 0.01 initial mutation probability
            n_best (): default = 2, how many of the best earthworm to keep from one generation to the next
            alpha (): default = 0.98, similarity factor
            beta (): default = 1, the initial proportional factor
            gamma (): default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.p_c = p_c
        self.p_m = p_m
        self.n_best = n_best
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        ## Dynamic variable
        self.dyn_beta = beta

    def create_child(self, idx, pop):
        ### Reproduction 1: the first way of reproducing
        x_t1 = self.problem.lb + self.problem.ub - self.alpha * pop[idx][self.ID_POS]

        ### Reproduction 2: the second way of reproducing
        if idx >= self.n_best:  ### Select two parents to mate and create two children
            idx = int(self.pop_size * 0.2)
            if np.random.uniform() < 0.5:  ## 80% parents selected from best population
                idx1, idx2 = np.random.choice(range(0, idx), 2, replace=False)
            else:  ## 20% left parents selected from worst population (make more diversity)
                idx1, idx2 = np.random.choice(range(idx, self.pop_size), 2, replace=False)
            r = np.random.uniform()
            x_child = r * pop[idx2][self.ID_POS] + (1 - r) * pop[idx1][self.ID_POS]
        else:
            r1 = np.random.randint(0, self.pop_size)
            x_child = pop[r1][self.ID_POS]
        x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
        pos_new = self.amend_position_faster(x_t1)
        fit_new = self.get_fitness_position(pos_new)
        if self.compare_agent([pos_new, fit_new], pop[idx]):
            return [pos_new, fit_new]
        return pop[idx].copy()

    def _updating_process(self, mode, pop, pop_elites, g_best):
        pos_list = np.array([item[self.ID_POS] for item in pop])
        x_mean = np.mean(pos_list, axis=0)
        ## Cauchy mutation (CM)
        cauchy_w = g_best[self.ID_POS].copy()
        for i in range(self.n_best, self.pop_size):  # Don't allow the elites to be mutated
            cauchy_w = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, x_mean, cauchy_w)
            x_t1 = (cauchy_w + g_best[self.ID_POS]) / 2
            pos_new = self.amend_position_faster(x_t1)
            pop[i][self.ID_POS] = pos_new
        pop = self.update_fitness_population(mode, pop)

        ## Elitism Strategy: Replace the worst with the previous generation's elites.
        pop, local_best = self.get_global_best_solution(pop)
        for i in range(0, self.n_best):
            pop[self.pop_size - i - 1] = pop_elites[i].copy()

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(pop):
            if tuple(obj[self.ID_POS].tolist()) in new_set:
                pop[idx] = self.create_solution()
            else:
                new_set.add(tuple(obj[self.ID_POS].tolist()))
        return pop

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
        ## Update the pop best
        pop_elites, local_best = self.get_global_best_solution(pop)
        pop_idx = np.array(range(0, self.pop_size))

        self.dyn_beta = self.gamma * self.beta
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            pop = [x for x in pop_child]
            pop = self._updating_process(mode, pop, pop_elites, g_best)
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop), pop_idx)
            pop = [x for x in pop_child]
            pop = self._updating_process(mode, pop, pop_elites, g_best)
        else:
            pop = [self.create_child(idx, pop) for idx in pop_idx]
            pop = self._updating_process(mode, pop, pop_elites, g_best)
        return pop

