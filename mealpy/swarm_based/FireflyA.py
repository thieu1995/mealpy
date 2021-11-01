#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:13, 01/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseFireflyA(Optimizer):
    """
        The original version of: Firefly Algorithm (FireflyA)
            Firefly Algorithm for Optimization Problem
        Link:
            DOI:
            https://www.researchgate.net/publication/259472546_Firefly_Algorithm_for_Optimization_Problem
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
            delta (float): Mutation Step Size, default = 0.05
            exponent (int): Exponent (m in the paper), default = 2
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = int(pop_size * (pop_size + 1) / 2 * 0.5)
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.gamma = gamma
        self.beta_base = beta_base
        self.alpha = alpha
        self.alpha_damp = alpha_damp
        self.delta = delta
        self.exponent = exponent

        ## Dynamic variable
        self.dyn_alpha = alpha   # Initial Value of Mutation Coefficient

    def create_child(self, idx, pop, dmax):
        agent = pop[idx].copy()
        for j in range(idx + 1, self.pop_size):
            # Move Towards Better Solutions
            if self.compare_agent(pop[j], agent):
                # Calculate Radius and Attraction Level
                rij = np.linalg.norm(agent[self.ID_POS] - pop[j][self.ID_POS]) / dmax
                beta = self.beta_base * np.exp(-self.gamma * rij ** self.exponent)

                # Mutation Vector
                mutation_vector = self.delta * np.random.uniform(0, 1, self.problem.n_dims)
                temp = np.matmul((pop[j][self.ID_POS] - agent[self.ID_POS]), np.random.uniform(0, 1, (self.problem.n_dims, self.problem.n_dims)))
                # print(temp)
                pos_new = agent[self.ID_POS] + self.dyn_alpha * mutation_vector + beta * temp

                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)

                # Compare to Previous Solution
                if self.compare_agent([pos_new, fit_new], agent):
                    agent = [pos_new, fit_new]
        return agent

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
        # Maximum Distance
        dmax = np.sqrt(self.problem.n_dims)
        pop_idx = np.array(range(0, self.pop_size-1))
        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, dmax=dmax), pop_idx)
            child = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop=pop, dmax=dmax), pop_idx)
            child = [x for x in pop_child]
        else:
            child = [self.create_child(idx, pop, dmax) for idx in pop_idx]
        child.append(g_best)

        # Merge, Sort and Selection, update global best
        # pop = self.get_sorted_strim_population(pop + child, self.pop_size)
        self.dyn_alpha = self.alpha_damp * self.alpha
        return child


