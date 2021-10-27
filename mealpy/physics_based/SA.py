#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 22:08, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseSA(Optimizer):
    """
        The original version of: Simulated Annealing (SA)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, max_sub_iter=10, t0=1000, t1=1, move_count=5,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (): Maximum Number of Sub-Iteration (within fixed temperature)
            t0 (): Initial Temperature
            t1 (): Final Temperature
            move_count (): Move Count per Individual Solution
            mutation_rate (): Mutation Rate
            mutation_step_size (): Mutation Step Size
            mutation_step_size_damp (): Mutation Step Size Damp
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size * max_sub_iter * move_count
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.max_sub_iter = max_sub_iter
        self.t0 = t0
        self.t1 = t1
        self.move_count = move_count
        self.mutation_rate = mutation_rate
        self.mutation_step_size = mutation_step_size
        self.mutation_step_size_damp = mutation_step_size_damp

    def mutate(self, position, sigma):
        mu = self.mutation_rate
        # Select Mutating Variables
        pos_new = position + sigma * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < mu, position, pos_new)

        if np.all(pos_new == position):  # Select at least one variable to mutate
            pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform()
        return self.amend_position_faster(pos_new)

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
        if mode != "sequential":
            print("SA is support sequential mode only!")
            exit(0)
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history.save_initial_best(g_best)

        # Initial Temperature
        t = self.t0  # Initial Temperature
        t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)  # Calculate Temperature Damp Rate
        sigma = self.mutation_step_size  # Initial Value of Step Size

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            # Sub-Iterations
            for g in range(0, self.max_sub_iter):

                # Create new population
                pop_new = []
                for i in range(0, self.pop_size):
                    for j in range(0, self.move_count):
                        # Perform Mutation (Move)
                        pos_new = self.mutate(pop[i][self.ID_POS], sigma)
                        pos_new = self.amend_position_faster(pos_new)
                        fit_new = self.get_fitness_position(pos_new)
                        pop_new.append([pos_new, fit_new])

                # Columnize and Sort Newly Created Population
                pop_new, g_best = self.update_global_best_solution(pop_new)
                pop_new = pop_new[:self.pop_size]

                # Randomized Selection
                for i in range(0, self.pop_size):
                    # Check if new solution is better than current
                    if self.compare_agent(pop_new[i], pop[i]):
                        pop[i] = pop_new[i].copy()
                    else:
                        # Compute difference according to problem type
                        delta = abs(pop_new[i][self.ID_FIT][self.ID_TAR] - pop[i][self.ID_FIT][self.ID_TAR])
                        p = np.exp(-delta / t)  # Compute Acceptance Probability
                        if np.random.uniform() <= p:  # Accept / Reject
                            pop[i] = pop_new[i].copy()
            # Update Temperature
            t = t_damp * t
            sigma = self.mutation_step_size_damp * sigma

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
