#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSFO(Optimizer):
    """
    The original version of: SailFish Optimizer (SFO)
    Link:
        https://doi.org/10.1016/j.engappai.2019.01.001
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pp=0.1, A=4, epxilon=0.0001, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            A (int): A = 4, 6,... (coefficient for decreasing the value of Power Attack linearly from A to 0)
            epxilon (float): should be 0.0001, 0.001
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pp = pp
        self.A = A
        self.epxilon = epxilon

        self.s_size = int(self.pop_size / self.pp)

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.s_pop = self.create_population(self.s_size)
        _, self.g_best = self.get_global_best_solution(self.pop)        # pop = sailfish
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)     # s_pop = sardines

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        nfe_epoch = 0
        pop_new = []
        PD = 1 - self.pop_size / (self.pop_size + self.s_size)
        for i in range(0, self.pop_size):
            lamda_i = 2 * np.random.uniform() * PD - PD
            pos_new = self.s_gbest[self.ID_POS] - lamda_i * (np.random.uniform() *
                                (self.pop[i][self.ID_POS] + self.s_gbest[self.ID_POS]) / 2 - self.pop[i][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
        nfe_epoch += self.pop_size

        ## Calculate AttackPower using Eq.(10)
        AP = self.A * (1 - 2 * (epoch + 1) * self.epxilon)
        if AP < 0.5:
            alpha = int(self.s_size * np.abs(AP))
            beta = int(self.problem.n_dims * np.abs(AP))
            ### Random np.random.choice number of sardines which will be updated their position
            list1 = np.random.choice(range(0, self.s_size), alpha)
            for i in range(0, self.s_size):
                if i in list1:
                    #### Random np.random.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                    pos_new = deepcopy(self.s_pop[i][self.ID_POS])
                    list2 = np.random.choice(range(0, self.problem.n_dims), beta, replace=False)
                    pos_new[list2] = (np.random.uniform(0, 1, self.problem.n_dims) *
                                      (self.pop[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP))[list2]
                    pos_new = self.amend_position_faster(pos_new)
                    self.s_pop[i] = [pos_new, None]
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, self.s_size):
                pos_new = np.random.uniform() * (self.g_best[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP)
                self.s_pop[i][self.ID_POS] = self.amend_position_faster(pos_new)
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_fitness_population(self.s_pop)
        nfe_epoch += self.s_size

        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop, g_best = self.get_global_best_solution(self.pop)
        self.s_pop, s_gbest = self.get_global_best_solution(self.s_pop)
        for i in range(0, self.pop_size):
            for j in range(0, self.s_size):
                ### If there is a better position in sardine population.
                if self.compare_agent(self.s_pop[j], self.pop[i]):
                    self.pop[i] = deepcopy(self.s_pop[j])
                    del self.s_pop[j]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        temp = self.s_size - len(self.s_pop)
        if temp == 1:
            self.s_pop = self.s_pop + [self.create_solution()]
        else:
            self.s_pop = self.s_pop + self.create_population(self.s_size - len(self.s_pop))
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)
        self.nfe_per_epoch = nfe_epoch


class ImprovedSFO(Optimizer):
    """
    My improved version of: Sailfish Optimizer (SFO)
    Notes:
        + Reform Energy equation,
        + No need parameter A and epxilon
        + Based on idea of Opposition-based Learning
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pp=0.1, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pp = pp

        self.s_size = int(self.pop_size / self.pp)

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.s_pop = self.create_population(self.s_size)
        _, self.g_best = self.get_global_best_solution(self.pop)
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        nfe_epoch = 0
        pop_new = []
        for i in range(0, self.pop_size):
            PD = 1 - len(self.pop) / (len(self.pop) + len(self.s_pop))
            lamda_i = 2 * np.random.uniform() * PD - PD
            pos_new = self.s_gbest[self.ID_POS] - lamda_i * (np.random.uniform() *
                        (self.g_best[self.ID_POS] + self.s_gbest[self.ID_POS]) / 2 - self.pop[i][self.ID_POS])
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
        nfe_epoch += self.pop_size

        ## ## Calculate AttackPower using my Eq.thieu
        #### This is our proposed, simple but effective, no need A and epxilon parameters
        AP = 1 - epoch * 1.0 / self.epoch
        if AP < 0.5:
            for i in range(0, len(self.s_pop)):
                temp = (self.g_best[self.ID_POS] + AP) / 2
                self.s_pop[i][self.ID_POS] = self.problem.lb + self.problem.ub - temp + \
                                             np.random.uniform() * (temp - self.s_pop[i][self.ID_POS])
        else:
            ### Update the position of all sardine using Eq.(9)
            for i in range(0, len(self.s_pop)):
                self.s_pop[i][self.ID_POS] = np.random.uniform() * (self.g_best[self.ID_POS] - self.s_pop[i][self.ID_POS] + AP)
        ## Recalculate the fitness of all sardine
        self.s_pop = self.update_fitness_population(self.s_pop)
        nfe_epoch += len(self.s_pop)

        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.pop = self.get_sorted_strim_population(self.pop, self.pop_size)
        self.s_pop = self.get_sorted_strim_population(self.s_pop, len(self.s_pop))
        for i in range(0, self.pop_size):
            for j in range(0, len(self.s_pop)):
                ### If there is a better position in sardine population.
                if self.compare_agent(self.s_pop[j], self.pop[i]):
                    self.pop[i] = deepcopy(self.s_pop[j])
                    del self.s_pop[j]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size

        self.s_pop = self.s_pop + self.create_population(self.s_size - len(self.s_pop))
        _, self.s_gbest = self.get_global_best_solution(self.s_pop)
        self.nfe_per_epoch = nfe_epoch
