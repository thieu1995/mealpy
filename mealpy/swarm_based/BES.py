#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
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

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, a=10, R=1.5,alpha=2, c1=2,c2=2):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.a = a           # default: 10, determining the corner between point search in the central point, in [5, 10]
        self.R = R           # default: 1.5, determining the number of search cycles, in [0.5, 2]
        self.alpha = alpha   # default: 2, parameter for controlling the changes in position, in [1.5, 2]
        self.c1 = c1         # default: 2, in [1, 2]
        self.c2 = c2         # c1 and c2 increase the movement intensity of bald eagles towards the best and centre points

    def _create_x_and_y__(self):
        ## Eq. 2
        phi = self.a * pi * uniform()
        r = phi + self.R * uniform()
        xr, yr = r * sin(phi), r * cos(phi)

        ## Eq. 3
        r1 = phi1 = self.a * pi * uniform()
        xr1, yr1 = r1 * sinh(phi1), r1 * cosh(phi1)
        return array([xr, yr, xr1, yr1])

    def _train__(self):
        # Initialization population and fitness
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## 0. Pre-definded
            xy_list = array([self._create_x_and_y__() for _ in range(0, self.pop_size)]).T
            x_list = xy_list[0] / max(xy_list[0])
            y_list = xy_list[1] / max(xy_list[1])
            x1_list = xy_list[2] / max(xy_list[2])
            y1_list = xy_list[3] / max(xy_list[3])

            # Three parts: selecting the search space, searching within the selected search space and swooping.

            ## 1. Select space
            solution_list = array([individual[self.ID_POS] for individual in pop])
            solution_mean = mean(solution_list, axis=0)
            for i in range(0, self.pop_size):
                temp = g_best[self.ID_POS] + self.alpha * uniform() * (solution_mean - pop[i][self.ID_POS])
                #temp = self._faster_amend_solution_and_return__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [temp, fit]

            ## 2. Search in space
            solution_list = array([individual[self.ID_POS] for individual in pop])
            solution_mean = mean(solution_list, axis=0)
            for i in range(0, self.pop_size):
                solution_i1 = pop[choice(range(0, self.pop_size))][self.ID_POS]
                temp = pop[i][self.ID_POS] + y_list[i] * (pop[i][self.ID_POS] - solution_i1) + x_list[i] * (pop[i][self.ID_POS] - solution_mean)
                #temp = self._faster_amend_solution_and_return__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [temp, fit]

            ## 3. Swoop
            solution_list = array([individual[self.ID_POS] for individual in pop])
            solution_mean = mean(solution_list, axis=0)
            for i in range(0, self.pop_size):
                temp = uniform() * g_best[self.ID_POS] + x1_list[i] * (pop[i][self.ID_POS] - self.c1 * solution_mean) + y1_list[i] * (pop[i][self.ID_POS] - self.c2 * g_best[self.ID_POS])
                #temp = self._faster_amend_solution_and_return__(temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]
                if fit < g_best[self.ID_FIT]:
                    g_best = [temp, fit]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
