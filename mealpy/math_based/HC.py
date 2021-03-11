#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 10:08, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import normal
from numpy import sum, mean, exp, array
from mealpy.root import Root


class OriginalHC(Root):
    """
    The original version of: Hill Climbing (HC)
    Noted:
        The number of neighbour solutions are equal to user defined
        The step size to calculate neighbour is randomized
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, neighbour_size=50, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.neighbour_size = neighbour_size

    def create_neighbor(self, position, step_size):
        pos_new = position + normal(0, 1, self.problem_size) * step_size
        pos_new = self.amend_position_faster(pos_new)
        return pos_new

    def train(self):
        g_best = self.create_solution()

        for epoch in range(self.epoch):
            step_size = mean(self.ub - self.lb) * exp(-2*(epoch+1) / self.epoch)
            pop_neighbours = []
            for i in range(0, self.neighbour_size):
                pos_new = self.create_neighbor(g_best[self.ID_POS], step_size)
                fit_new = self.get_fitness_position(pos_new)
                pop_neighbours.append([pos_new, fit_new])
            pop_neighbours.append(g_best)
            g_best = self.get_global_best_solution(pop_neighbours, self.ID_FIT, self.ID_MIN_PROB)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseHC(Root):
    """
    The modified version of: Hill Climbing (HC) based on swarm-of people are trying to climb on the mountain ideas
    Noted:
        The number of neighbour solutions are equal to population size
        The step size to calculate neighbour is randomized and based on ranks of solution.
            + The guys near on top of mountain will move slower than the guys on bottom of mountain.
            + Imagine it is like: exploration when far from global best, and exploitation when near global best
        Who on top of mountain first will be the winner. (global optimal)
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, neighbour_size=50, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.neighbour_size = neighbour_size

    def create_neighbor(self, position, step_size):
        pos_new = position + normal(0, 1, self.problem_size) * step_size
        pos_new = self.amend_position_faster(pos_new)
        return pos_new

    def train(self):

        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        ranks = array(list(range(1, self.pop_size+1)))
        ranks = ranks / sum(ranks)

        for epoch in range(self.epoch):
            step_size = mean(self.ub - self.lb) * exp(-2 * (epoch + 1) / self.epoch)

            for i in range(0, self.pop_size):
                ss = step_size * ranks[i]
                pop_neighbours = []
                for j in range(0, self.neighbour_size):
                    pos_new = self.create_neighbor(pop[i][self.ID_POS], ss)
                    fit_new = self.get_fitness_position(pos_new)
                    pop_neighbours.append([pos_new, fit_new])
                pop_neighbours.append(g_best)
                pop[i] = self.get_global_best_solution(pop_neighbours, self.ID_FIT, self.ID_MIN_PROB)

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

