#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Update the pop best
        pop_elites, local_best = self.get_global_best_solution(self.pop)
        nfe_epoch = 0
        pop = []
        for idx in range(0, self.pop_size):
            ### Reproduction 1: the first way of reproducing
            x_t1 = self.problem.lb + self.problem.ub - self.alpha * self.pop[idx][self.ID_POS]

            ### Reproduction 2: the second way of reproducing
            if idx >= self.n_best:  ### Select two parents to mate and create two children
                idx = int(self.pop_size * 0.2)
                if np.random.uniform() < 0.5:  ## 80% parents selected from best population
                    idx1, idx2 = np.random.choice(range(0, idx), 2, replace=False)
                else:  ## 20% left parents selected from worst population (make more diversity)
                    idx1, idx2 = np.random.choice(range(idx, self.pop_size), 2, replace=False)
                r = np.random.uniform()
                x_child = r * self.pop[idx2][self.ID_POS] + (1 - r) * self.pop[idx1][self.ID_POS]
            else:
                r1 = np.random.randint(0, self.pop_size)
                x_child = self.pop[r1][self.ID_POS]
            x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
            pos_new = self.amend_position_faster(x_t1)
            pop.append([pos_new, None])
        pop = self.update_fitness_population(pop)
        pop = self.greedy_selection_population(self.pop, pop)
        nfe_epoch += self.pop_size
        self.dyn_beta = self.gamma * self.beta

        pos_list = np.array([item[self.ID_POS] for item in pop])
        x_mean = np.mean(pos_list, axis=0)
        ## Cauchy mutation (CM)
        cauchy_w = deepcopy(self.g_best[self.ID_POS])
        for i in range(self.n_best, self.pop_size):  # Don't allow the elites to be mutated
            cauchy_w = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, x_mean, cauchy_w)
            x_t1 = (cauchy_w + self.g_best[self.ID_POS]) / 2
            pos_new = self.amend_position_faster(x_t1)
            pop[i][self.ID_POS] = pos_new
            nfe_epoch += 1
        pop = self.update_fitness_population(pop)

        ## Elitism Strategy: Replace the worst with the previous generation's elites.
        pop, local_best = self.get_global_best_solution(pop)
        for i in range(0, self.n_best):
            pop[self.pop_size - i - 1] = deepcopy(pop_elites[i])

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(pop):
            if tuple(obj[self.ID_POS].tolist()) in new_set:
                pop[idx] = self.create_solution()
                nfe_epoch += 1
            else:
                new_set.add(tuple(obj[self.ID_POS].tolist()))
        self.nfe_per_epoch = nfe_epoch
        self.pop = pop

