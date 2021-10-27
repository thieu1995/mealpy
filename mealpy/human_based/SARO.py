#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:16, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseSARO(Optimizer):
    """
    My version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    Notes:
        + Remove all third loop
    """
    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 3 * pop_size

        self.epoch = epoch
        self.pop_size = pop_size
        self.se = se
        self.mu = mu

        ## Dynamic variable
        self.dyn_USN = np.zeros(self.pop_size)

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
        pop = self.create_population(mode, 2*self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)
        self.history.save_initial_best(g_best)

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            ## Evolve method will be called in child class
            pop = self.evolve(mode, epoch, pop, g_best)

            # update global best position
            pop, g_best = self.update_global_best_solution(pop)  # We sort the population

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
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
            print("SARO algorithm is support sequential mode only!")
            exit(0)

        pop_x = pop[:self.pop_size].copy()
        pop_m = pop[self.pop_size:].copy()

        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - pop[k][self.ID_POS]

            #### Remove third loop here, also using random flight back when out of bound
            pos_new_1 = pop[k][self.ID_POS] + np.random.uniform() * sd
            pos_new_2 = pop_x[idx][self.ID_POS] + np.random.uniform() * sd
            pos_new = np.where(np.logical_and(np.random.uniform(0, 1, self.problem.n_dims) < self.se,
                                              pop[k][self.ID_FIT] < pop_x[idx][self.ID_FIT]), pos_new_1, pos_new_2)
            pos_new = self.amend_position_random(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = [pos_new, fit_new]
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            ## Individual phase
            pop = pop_x.copy() + pop_m.copy()
            k1, k2 = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)

            #### Remove third loop here, and flight back strategy now be a random
            pos_new = g_best[self.ID_POS] + np.random.uniform() * (pop[k1][self.ID_POS] - pop[k2][self.ID_POS])
            pos_new = self.amend_position_random(pos_new)
            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = [pos_new, fit_new]
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution()
                self.dyn_USN[idx] = 0
            return (pop_x + pop_m)


class OriginalSARO(BaseSARO):
    """
    The original version of: Search And Rescue Optimization (SAR)
        (A New Optimization Algorithm Based on Search and Rescue Operations)
    Link:
        https://doi.org/10.1155/2019/2482543
    """

    def __init__(self, problem, epoch=10000, pop_size=100, se=0.5, mu=50, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 50
        """
        super().__init__(problem, epoch, pop_size, se, mu, **kwargs)

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
            print("SARO algorithm is support sequential mode only!")
            exit(0)

        pop_x = pop[:self.pop_size].copy()
        pop_m = pop[self.pop_size:].copy()

        for idx in range(self.pop_size):
            ## Social Phase
            k = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx][self.ID_POS] - pop[k][self.ID_POS]
            j_rand = np.random.randint(0, self.problem.n_dims)
            r1 = np.random.uniform(-1, 1)

            pos_new = pop_x[idx][self.ID_POS].copy()
            for j in range(0, self.problem.n_dims):
                if np.random.uniform() < self.se or j == j_rand:
                    if self.compare_agent(pop[k], pop_x[idx]):
                        pos_new[j] = pop[k][self.ID_POS][j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[idx][self.ID_POS][j] + r1 * sd[j]

                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2
            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = [pos_new, fit_new]
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            ## Individual phase
            pop = pop_x.copy() + pop_m.copy()

            k, m = np.random.choice(list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False)
            pos_new = pop_x[idx][self.ID_POS] + np.random.uniform() * (pop[k][self.ID_POS] - pop[m][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx][self.ID_POS][j] + self.problem.ub[j]) / 2

            fit_new = self.get_fitness_position(pos_new)
            if self.compare_agent([pos_new, fit_new], pop_x[idx]):
                pop_m[np.random.randint(0, self.pop_size)] = pop_x[idx]
                pop_x[idx] = [pos_new, fit_new]
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.create_solution()
                self.dyn_USN[idx] = 0
            return pop_x.copy() + pop_m.copy()
