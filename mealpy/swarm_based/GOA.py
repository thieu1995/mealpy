#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:53, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, zeros, sqrt, remainder, sum
from numpy.random import normal
from mealpy.root import Root


class BaseGOA(Root):
    """
    The original version of: Grasshopper Optimization Algorithm (GOA)
        (Grasshopper Optimisation Algorithm: Theory and Application Advances in Engineering Software)
    Link:
        http://dx.doi.org/10.1016/j.advengsoft.2017.01.004
        https://www.mathworks.com/matlabcentral/fileexchange/61421-grasshopper-optimisation-algorithm-goa
    Notes:
        + I added normal() component to Eq, 2.7
        + Changed the way to calculate distance between two location
        + Used batch-size idea
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, c_minmax=(0.00004, 1), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c_minmax = c_minmax    # coefficient c

    def _s_function__(self, r_vector=None):
        f = 0.5
        l = 1.5
        return f * exp(-r_vector / l) - exp(-r_vector)        # Eq.(2.3) in the paper

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Eq.(2.8) in the paper
            c = self.c_minmax[1] - epoch * ((self.c_minmax[1] - self.c_minmax[0]) / self.epoch)
            for i in range(0, self.pop_size):
                S_i_total = zeros(self.problem_size)
                for j in range(0, self.pop_size):
                    dist = sqrt(sum((pop[i][self.ID_POS] - pop[j][self.ID_POS])**2))
                    r_ij_vector = (pop[i][self.ID_POS] - pop[j][self.ID_POS]) / (dist + self.EPSILON)    # xj - xi / dij in Eq.(2.7)
                    xj_xi = 2 + remainder(dist, 2)  # |xjd - xid| in Eq. (2.7)
                    ## The first part inside the big bracket in Eq. (2.7)   16 955 230 764    212 047 193 643
                    ran = (c / 2) * (self.ub - self.lb)
                    s_ij = ran * self._s_function__(xj_xi) * r_ij_vector
                    S_i_total += s_ij

                x_new = c * normal() * S_i_total + g_best[self.ID_POS]     # Eq. (2.7) in the paper
                x_new = self.amend_position_faster(x_new)
                fit = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit]

                # batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
