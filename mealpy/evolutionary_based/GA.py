#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
        Genetic Algorithm (GA)
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pc=0.95, pm=0.025, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 2 * pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

    def create_child(self, agent_i, pop_copy, list_fitness):
        ### Selection
        # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
        id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
        id_c2 = self.get_index_roulette_wheel_selection(list_fitness)

        w1 = pop_copy[id_c1][self.ID_POS]
        w2 = pop_copy[id_c2][self.ID_POS]
        ### Crossover
        if np.random.uniform() < self.pc:
            w1, w2 = self.crossover_arthmetic_recombination(w1, w2)

        ### Mutation, remove third loop here
        w1 = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, np.random.uniform(self.problem.lb, self.problem.ub), w1)
        w2 = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, np.random.uniform(self.problem.lb, self.problem.ub), w2)

        if np.random.uniform() < 0.5:
            return [w1.copy(), self.get_fitness_position(w1)]
        else:
            return [w2.copy(), self.get_fitness_position(w2)]

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
        # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
        list_fitness = np.array([agent[self.ID_FIT][self.ID_TAR] for agent in pop])
        pop_copy = pop.copy()

        if mode == "thread":
            with parallel.ThreadPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, list_fitness=list_fitness), pop)
            pop = [x for x in pop_child]
        elif mode == "process":
            with parallel.ProcessPoolExecutor() as executor:
                pop_child = executor.map(partial(self.create_child, pop_copy=pop_copy, list_fitness=list_fitness), pop)
            pop = [x for x in pop_child]
        else:
            pop = [self.create_child(agent, pop_copy, list_fitness) for agent in pop]
        return pop
