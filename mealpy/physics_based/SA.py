#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 22:08, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseSA(Optimizer):
    """
        The original version of: Simulated Annealing (SA)
    """

    def __init__(self, problem, epoch=10000, pop_size=100, max_sub_iter=5, t0=1000, t1=1, move_count=5,
                 mutation_rate=0.1, mutation_step_size=0.1, mutation_step_size_damp=0.99, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (int): Maximum Number of Sub-Iteration (within fixed temperature)
            t0 (int): Initial Temperature
            t1 (int): Final Temperature
            move_count (int): Move Count per Individual Solution
            mutation_rate (float): Mutation Rate
            mutation_step_size (float): Mutation Step Size
            mutation_step_size_damp (float): Mutation Step Size Damp
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

        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def _mutate(self, position, sigma):
        # Select Mutating Variables
        pos_new = position + sigma * np.random.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.mutation_rate, position, pos_new)

        if np.all(pos_new == position):  # Select at least one variable to _mutate
            pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform()
        return self.amend_position_faster(pos_new)

    def initialization(self):
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        self.pop = self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):

            # Create new population
            pop_new = []
            for i in range(0, self.pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    pos_new = self._mutate(self.pop[i][self.ID_POS], self.dyn_sigma)
                    pos_new = self.amend_position_faster(pos_new)
                    pop_new.append([pos_new, None])
            pop_new = self.update_fitness_population(pop_new)

            # Columnize and Sort Newly Created Population
            pop_new = self.get_sorted_strim_population(pop_new, self.pop_size)

            # Randomized Selection
            for i in range(0, self.pop_size):
                # Check if new solution is better than current
                if self.compare_agent(pop_new[i], self.pop[i]):
                    self.pop[i] = deepcopy(pop_new[i])
                else:
                    # Compute difference according to problem type
                    delta = abs(pop_new[i][self.ID_FIT][self.ID_TAR] - self.pop[i][self.ID_FIT][self.ID_TAR])
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if np.random.uniform() <= p:  # Accept / Reject
                        self.pop[i] = deepcopy(pop_new[i])
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma
