#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 15:37, 19/03/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from mealpy.root import Root
from numpy import array, mean, exp, ones
from numpy.random import rand, normal


class OriginalHGS(Root):
    """
        The original version of: Hunger Games Search (HGS)
        Link:
            https://aliasgharheidari.com/HGS.html
            Hunger Games Search (HGS): Visions, Conception, Implementation, Deep Analysis, Perspectives, and Towards Performance Shifts
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, L=0.08, LH=10000, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.L = L          # Switching updating  position probability
        self.LH = LH        # Largest hunger / threshold

    def get_hunger_list(self, pop=None, hunger_list=array, g_best=None, g_worst=None):
        # min_index = pop.index(min(pop, key=lambda x: x[self.ID_FIT]))
        # Eq (2.8) and (2.9)
        for i in range(0, self.pop_size):
            r = rand()
            # space: since we pass lower bound and upper bound as list. Better take the mean of them.
            space = mean(self.ub - self.lb)
            H = (pop[i][self.ID_FIT] - g_best[self.ID_FIT]) / (g_worst[self.ID_FIT] - g_best[self.ID_FIT] + self.EPSILON) * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            hunger_list[i] += H

            if g_best[self.ID_FIT] == pop[i][self.ID_FIT]:
                hunger_list[i] = 0
        return hunger_list

    def sech(self, x):
        return 2 / (exp(x) + exp(-x))

    def train(self):
        # Hungry value of all solutions
        hunger_list = ones(self.pop_size)

        # Create population
        pop = [self.create_solution() for _ in range(self.pop_size)]

        ## Eq. (2.2)
        ### Find the current best and current worst
        g_best, g_worst = self.get_global_best_global_worst_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        hunger_list = self.get_hunger_list(pop, hunger_list, g_best, g_worst)

        # Loop
        for epoch in range(self.epoch):

            ## Eq. (2.4)
            shrink = 2 * (1 - (epoch + 1) / self.epoch)

            for i in range(0, self.pop_size):
                #### Variation control
                E = self.sech(pop[i][self.ID_FIT] - g_best[self.ID_FIT])

                # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
                R = 2 * shrink * rand() - shrink  # Eq. (2.3)

                ## Calculate the hungry weight of each position
                if rand() < self.L:
                    W1 = hunger_list[i] * self.pop_size / (sum(hunger_list) + self.EPSILON) * rand()
                else:
                    W1 = 1
                W2 = (1 - exp(-abs(hunger_list[i] - sum(hunger_list)))) * rand() * 2

                ### Udpate position of individual Eq. (2.1)
                r1 = rand()
                r2 = rand()
                if r1 < self.L:
                    pos_new = pop[i][self.ID_POS] * (1 + normal(0, 1))
                else:
                    if r2 > E:
                        pos_new = W1 * g_best[self.ID_POS] + R * W2 * abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    else:
                        pos_new = W1 * g_best[self.ID_POS] - R * W2 * abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                fit_new = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Update global best and global worst
            g_best, g_worst = self.update_global_best_global_worst_solution(pop, self.ID_MIN_PROB, self.ID_MAX_PROB, g_best)

            ## Update hunger list
            hunger_list = self.get_hunger_list(pop, hunger_list, g_best, g_worst)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

