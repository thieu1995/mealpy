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
from copy import deepcopy
from mealpy.root import Root


class BaseGA(Root):
    """
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """
    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, pc=0.95, pm=0.025, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

    def _create_next_generation__(self, pop):
        next_population = []
        while (len(next_population) < self.pop_size):
            ### Selection
            # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
            fitness_list = array([item[self.ID_FIT] for item in pop])
            id_c1 = self.get_index_roulette_wheel_selection(fitness_list)
            id_c2 = self.get_index_roulette_wheel_selection(fitness_list)

            w1 = pop[id_c1][self.ID_POS]
            w2 = pop[id_c2][self.ID_POS]
            ### Crossover
            if uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(w1, w2)

            ### Mutation, remove third loop here
            w1 = where(uniform(0, 1, self.problem_size) < self.pm, uniform(self.lb, self.ub, self.problem_size), w1)
            w2 = where(uniform(0, 1, self.problem_size) < self.pm, uniform(self.lb, self.ub, self.problem_size), w2)

            c1_new = [deepcopy(w1), self.get_fitness_position(w1)]
            c2_new = [deepcopy(w2), self.get_fitness_position(w2)]
            next_population.append(c1_new)
            next_population.append(c2_new)
        return next_population

    def train(self):
        pop = [self.create_solution(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Next generations
            pop = deepcopy(self._create_next_generation__(pop))

            # update global best position
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
