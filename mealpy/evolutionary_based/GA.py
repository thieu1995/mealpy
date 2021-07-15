#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
        Genetic Algorithm (GA)
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """

    def __init__(self, problem: dict, epoch=1000, pop_size=100, pc=0.95, pm=0.025):
        """
        Args:
            problem (dict): a dictionary of your problem
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
        """
        super().__init__(problem)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        _, g_best = self.get_global_best_solution(pop)                  # We don't sort the population
        self.history_list_g_best = [g_best]
        self.history_list_c_best = self.history_list_g_best.copy()

        for epoch in range(0, self.epoch):
            time_start = time.time()

            # Next generations
            next_population = []
            while (len(next_population) < self.pop_size):
                ### Selection
                # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
                fitness_list = np.array([agent[self.ID_FIT][self.ID_TAR] for agent in pop])
                id_c1 = self.get_index_roulette_wheel_selection(fitness_list)
                id_c2 = self.get_index_roulette_wheel_selection(fitness_list)

                w1 = pop[id_c1][self.ID_POS]
                w2 = pop[id_c2][self.ID_POS]
                ### Crossover
                if np.random.uniform() < self.pc:
                    w1, w2 = self.crossover_arthmetic_recombination(w1, w2)

                ### Mutation, remove third loop here
                w1 = np.where(np.random.uniform(0, 1, self.problem_size) < self.pm, np.random.uniform(self.lb, self.ub), w1)
                w2 = np.where(np.random.uniform(0, 1, self.problem_size) < self.pm, np.random.uniform(self.lb, self.ub), w2)

                c1_new = [w1.copy(), self.get_fitness_position(w1)]
                c2_new = [w2.copy(), self.get_fitness_position(w2)]
                next_population.append(c1_new)
                next_population.append(c2_new)

            pop = next_population.copy()
            # update global best position
            self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_start = time.time() - time_start
            self.history_list_epoch_time.append(time_start)
            self.print_epoch(epoch+1, time_start)
            self.history_list_pop.append(pop.copy())

        ## Additional information for the framework
        self.solution = self.history_list_g_best[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]
