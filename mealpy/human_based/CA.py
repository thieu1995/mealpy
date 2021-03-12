#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 12:09, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import array, min, max
from mealpy.root import Root


class OriginalCA(Root):
    """
    The original version of: Culture Algorithm (CA)
        Based on Ruby version in the book: Clever Algorithm (Jason Brown)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, accepted_rate=0.2, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.accepted_rate = accepted_rate

    def binary_tournament(self, population):
        id1, id2 = choice(list(range(0, len(population))), 2, replace=False)
        return population[id1] if (population[id1][self.ID_FIT] < population[id2][self.ID_FIT]) else population[id2]

    def create_faithful(self, lb, ub):
        position = uniform(lb, ub)
        fitness = self.get_fitness_position(position=position)
        return [position, fitness]

    def update_belief_space(self, belief_space, pop_accepted):
        pos_list = array([solution[self.ID_POS] for solution in pop_accepted])
        belief_space["lb"] = min(pos_list, axis=0)
        belief_space["ub"] = max(pos_list, axis=0)
        return belief_space

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        belief_space = {
            "lb": self.lb,
            "ub": self.ub,
        }
        accepted_num = int(self.accepted_rate * self.pop_size)
        # update situational knowledge (g_best here is a element inside belief space)

        for epoch in range(self.epoch):

            # create next generation
            pop_child = [self.create_faithful(belief_space["lb"], belief_space["ub"]) for _ in range(0, self.pop_size)]

            # select next generation
            pop = [self.binary_tournament(pop + pop_child) for _ in range(0, self.pop_size)]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)

            # Get accepted faithful
            accepted = pop[:accepted_num]

            # Update belief_space
            belief_space = self.update_belief_space(belief_space, accepted)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
