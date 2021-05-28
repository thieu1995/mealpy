#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 11:38, 02/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import random
from numpy import exp, where
from mealpy.root import Root


class BaseSalpSO(Root):
    """
    The original version of: Salp Swarm Optimization (SalpSO)
    Link:
        https://doi.org/10.1016/j.advengsoft.2017.07.002
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        # Initial population and find the global best solution
        pop = [self.create_solution() for _ in range(0, self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Main loop
        for epoch in range(0, self.epoch):

            c1 = 2 * exp(-((4 * (epoch+1) / self.epoch) ** 2))
            ## Eq. (3.2) in the paper
            for i in range(0, self.pop_size):

                if i < self.pop_size / 2:
                    c2_list = random(self.problem_size)
                    c3_list = random(self.problem_size)
                    pos_new_1 = g_best[self.ID_POS] + c1 * ((self.ub - self.lb) * c2_list + self.lb)
                    pos_new_2 = g_best[self.ID_POS] - c1 * ((self.ub - self.lb) * c2_list + self.lb)
                    pos_new = where(c3_list < 0.5, pos_new_1, pos_new_2)
                else:
                    # Eq. (3.4) in the paper
                    pos_new = (pop[i][self.ID_POS] + pop[i-1][self.ID_POS]) / 2
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

            # Check if salps go out of the search space and bring it back then re-calculate its fitness value
            # for i in range(0, self.pop_size):
            #     pos_new = self.amend_position_faster(pop[i][self.ID_POS])
            #     fit_new = self.get_fitness_position(pos_new)
            #     pop[i] = [pos_new, fit_new]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
