#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 18:37, 28/05/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import choice, uniform
from mealpy.root import Root


class BaseCSA(Root):
    """
        The original version of: Cuckoo Search Algorithm (CSA)
            (Cuckoo search via Levy flights)
        Link:
            https://doi.org/10.1109/NABIC.2009.5393690
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, p_a=0.3, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.p_a = p_a

    def train(self):
        n_cut = int(self.p_a * self.pop_size)
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):

            for i in range(0, self.pop_size):
                ## Generate levy-flight solution
                # pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS], step=0.001, case=2)
                pos_new = pop[i][self.ID_POS] + self.step_size_by_levy_flight(multiplier=0.001, case=-1)
                fit_new = self.get_fitness_position(pos_new)

                j_idx = choice(list(set(range(0, self.pop_size)) - {i}))
                if fit_new < pop[j_idx][self.ID_FIT]:
                    pop[j_idx] = [pos_new, fit_new]

            ## Abandoned some worst nests
            pop = sorted(pop, key=lambda item: item[self.ID_FIT], reverse=True)
            for i in range(0, n_cut):
                # pos_new = self.levy_flight(epoch, pop[i][self.ID_POS], g_best[self.ID_POS])
                pos_new = self.step_size_by_levy_flight(multiplier=0.001, case=-1)
                # pos_new = uniform(self.lb, self.ub, self.problem_size)
                fit_new = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

