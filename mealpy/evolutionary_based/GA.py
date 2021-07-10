#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, where
from numpy.random import uniform
from time import time
from mealpy.root import Root


class BaseGA(Root):
    """
        Genetic Algorithm (GA)
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """

    def __init__(self, obj_func=None, lb=None, ub=None, minmax='min', verbose=True, epoch=750, pop_size=100, pc=0.95, pm=0.025, **kwargs):
        super().__init__(obj_func, lb, ub, minmax, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        self.g_best_list = [self.get_global_best_solution(pop) ]
        self.c_best_list = self.g_best_list.copy()

        for epoch in range(0, self.epoch):
            time_start = time()

            # Next generations
            next_population = []
            while (len(next_population) < self.pop_size):
                ### Selection
                # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
                fitness_list = array([agent[self.ID_FIT][self.ID_TAR] for agent in pop])
                id_c1 = self.get_index_roulette_wheel_selection(fitness_list)
                id_c2 = self.get_index_roulette_wheel_selection(fitness_list)

                w1 = pop[id_c1][self.ID_POS]
                w2 = pop[id_c2][self.ID_POS]
                ### Crossover
                if uniform() < self.pc:
                    w1, w2 = self.crossover_arthmetic_recombination(w1, w2)

                ### Mutation, remove third loop here
                w1 = where(uniform(0, 1, self.problem_size) < self.pm, uniform(self.lb, self.ub), w1)
                w2 = where(uniform(0, 1, self.problem_size) < self.pm, uniform(self.lb, self.ub), w2)

                c1_new = [w1.copy(), self.get_fitness_position(w1)]
                c2_new = [w2.copy(), self.get_fitness_position(w2)]
                next_population.append(c1_new)
                next_population.append(c2_new)

            pop = next_population.copy()
            # update global best position
            self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_start = time() - time_start
            self.epoch_time_list.append(time_start)
            self.print_epoch(epoch+1, time_start)
            self.pop_list.append(pop.copy())

        ## Additional information for the framework
        self.solution = self.g_best_list[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR], self.g_best_list, self.c_best_list
