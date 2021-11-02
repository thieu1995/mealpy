#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%


import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pp = pp
        self.A = A
        self.epxilon = epxilon

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        self.termination_start()
        s_size = int(self.pop_size / self.pp)
        sf_pop = self.create_population(mode, self.pop_size)
        s_pop = self.create_population(mode, s_size)
        _, sf_gbest = self.get_global_best_solution(sf_pop)
        _, s_gbest = self.get_global_best_solution(s_pop)
        self.history.save_initial_best(sf_gbest)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / (len(sf_pop) + len(s_pop))
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * \
                    (np.random.uniform() * (sf_gbest[self.ID_POS] + s_gbest[self.ID_POS]) / 2 - sf_pop[i][self.ID_POS])

            ## Calculate AttackPower using Eq.(10)
            AP = self.A * (1 - 2 * (epoch + 1) * self.epxilon)
            if AP < 0.5:
                alpha = int(len(s_pop) * np.abs(AP))
                beta = int(self.problem.n_dims * np.abs(AP))
                ### Random np.random.choice number of sardines which will be updated their position
                list1 = np.random.choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        #### Random np.random.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                        list2 = np.random.choice(range(0, self.problem.n_dims), beta, replace=False)
                        s_pop[i][self.ID_POS][list2] = (np.random.uniform(0, 1, self.problem.n_dims) *
                                                        (sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP))[list2]
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = np.random.uniform() * (sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP)

            ## Recalculate the fitness of all sardine
            s_pop = self.update_fitness_population(mode, s_pop)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = self.get_sorted_strim_population(sf_pop, len(sf_pop))
            s_pop = self.get_sorted_strim_population(sf_pop, len(s_pop))
            for i in range(0, self.pop_size):
                for j in range(0, len(s_pop)):
                    ### If there is a better position in sardine population.
                    if self.compare_agent(s_pop[j], sf_pop[i]):
                        sf_pop[i] = s_pop[j].copy()
                        del s_pop[j]
                    break  #### This simple keyword helped reducing ton of comparing operation.
                    #### Especially when sardine pop size >> sailfish pop size

            s_pop = s_pop + self.create_population(mode, s_size - len(s_pop))
            _, sf_gbest = self.get_global_best_solution(sf_pop)
            _, s_gbest = self.get_global_best_solution(s_pop)

            _, g_best = self.update_global_best_solution([sf_gbest, s_gbest])

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(sf_pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]



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
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pp = pp

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        self.termination_start()
        s_size = int(self.pop_size / self.pp)
        sf_pop = self.create_population(mode, self.pop_size)
        s_pop = self.create_population(mode, s_size)
        _, sf_gbest = self.get_global_best_solution(sf_pop)
        _, s_gbest = self.get_global_best_solution(s_pop)
        self.history.save_initial_best(sf_gbest)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, self.pop_size):
                PD = 1 - len(sf_pop) / (len(sf_pop) + len(s_pop))
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop[i][self.ID_POS] = s_gbest[self.ID_POS] - lamda_i * \
                        (np.random.uniform() * (sf_gbest[self.ID_POS] + s_gbest[self.ID_POS]) / 2 - sf_pop[i][self.ID_POS])

            ## ## Calculate AttackPower using my Eq.thieu
            #### This is our proposed, simple but effective, no need A and epxilon parameters
            AP = 1 - epoch * 1.0 / self.epoch
            if AP < 0.5:
                for i in range(0, len(s_pop)):
                    temp = (sf_gbest[self.ID_POS] + AP) / 2
                    s_pop[i][self.ID_POS] = self.problem.lb + self.problem.ub - temp + np.random.uniform() * (temp - s_pop[i][self.ID_POS])
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop[i][self.ID_POS] = np.random.uniform() * (sf_gbest[self.ID_POS] - s_pop[i][self.ID_POS] + AP)

            ## Recalculate the fitness of all sardine
            s_pop = self.update_fitness_population(mode, s_pop)

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = self.get_sorted_strim_population(sf_pop, len(sf_pop))
            s_pop = self.get_sorted_strim_population(sf_pop, len(s_pop))
            for i in range(0, self.pop_size):
                for j in range(0, len(s_pop)):
                    ### If there is a better position in sardine population.
                    if self.compare_agent(s_pop[j], sf_pop[i]):
                        sf_pop[i] = s_pop[j].copy()
                        del s_pop[j]
                    break  #### This simple keyword helped reducing ton of comparing operation.
                    #### Especially when sardine pop size >> sailfish pop size

            s_pop = s_pop + self.create_population(mode, s_size - len(s_pop))
            _, sf_gbest = self.get_global_best_solution(sf_pop)
            _, s_gbest = self.get_global_best_solution(s_pop)

            _, g_best = self.update_global_best_solution([sf_gbest, s_gbest])

            ## Additional information for the framework
            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(sf_pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]


