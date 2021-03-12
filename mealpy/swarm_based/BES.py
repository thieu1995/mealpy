#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, choice
from numpy import array, mean, pi, sin, cos, max, sinh, cosh
from mealpy.root import Root


class BaseBES(Root):
    """
    Original version of: Bald Eagle Search (BES)
        (Novel meta-heuristic bald eagle search optimisation algorithm)
    Link:
        DOI: https://doi.org/10.1007/s10462-019-09732-5
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 a=10, R=1.5, alpha=2, c1=2, c2=2, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs=kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.a = a           # default: 10, determining the corner between point search in the central point, in [5, 10]
        self.R = R           # default: 1.5, determining the number of search cycles, in [0.5, 2]
        self.alpha = alpha   # default: 2, parameter for controlling the changes in position, in [1.5, 2]
        self.c1 = c1         # default: 2, in [1, 2]
        self.c2 = c2         # c1 and c2 increase the movement intensity of bald eagles towards the best and centre points

    def _create_x_y_x1_y1_(self):
        """ Using numpy vector for faster computational time """
        ## Eq. 2
        phi = self.a * pi * uniform(0, 1, self.pop_size)
        r = phi + self.R * uniform(0, 1, self.pop_size)
        xr, yr = r * sin(phi), r * cos(phi)

        ## Eq. 3
        r1 = phi1 = self.a * pi * uniform(0, 1, self.pop_size)
        xr1, yr1 = r1 * sinh(phi1), r1 * cosh(phi1)

        x_list = xr / max(xr)
        y_list = yr / max(yr)
        x1_list = xr1 / max(xr1)
        y1_list = yr1 / max(yr1)
        return x_list, y_list, x1_list, y1_list

    def train(self):
        # Initialization population and fitness
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## 0. Pre-definded
            x_list, y_list, x1_list, y1_list = self._create_x_y_x1_y1_()

            # Three parts: selecting the search space, searching within the selected search space and swooping.
            ## 1. Select space
            pos_list = array([individual[self.ID_POS] for individual in pop])
            pos_mean = mean(pos_list, axis=0)
            for i in range(0, self.pop_size):
                pos_new = g_best[self.ID_POS] + self.alpha * uniform() * (pos_mean - pop[i][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
                    if pop[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = [pos_new, fit]

            ## 2. Search in space
            pos_list = array([individual[self.ID_POS] for individual in pop])
            pos_mean = mean(pos_list, axis=0)
            for i in range(0, self.pop_size):
                idx_rand = choice(list(set(range(0, self.pop_size)) - {i}))
                pos_new = pop[i][self.ID_POS] + y_list[i] * (pop[i][self.ID_POS] - pop[idx_rand][self.ID_POS]) + x_list[i] * (pop[i][self.ID_POS] - pos_mean)
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
                    if pop[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = [pos_new, fit]

            ## 3. Swoop
            pos_list = array([individual[self.ID_POS] for individual in pop])
            pos_mean = mean(pos_list, axis=0)
            for i in range(0, self.pop_size):
                pos_new = uniform() * g_best[self.ID_POS] + x1_list[i] * (pop[i][self.ID_POS] - self.c1 * pos_mean) \
                          + y1_list[i] * (pop[i][self.ID_POS] - self.c2 * g_best[self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
                    if pop[i][self.ID_FIT] < g_best[self.ID_FIT]:
                        g_best = [pos_new, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
