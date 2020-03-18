#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:44, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import pi, sin, abs, cos
from copy import deepcopy
from mealpy.root import Root


class BaseSCA(Root):
    """
    This is my version of SCA. The original version of SCA in the above, it cannot convergence at all.
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Update the position of solutions with respect to destination
            for i in range(self.pop_size):              # i-th solution
                # Eq 3.4, r1 decreases linearly from a to 0
                a = 2.0
                r1 = a - (epoch + 1) * (a / self.epoch)
                temp = deepcopy(pop[i][self.ID_POS])
                for j in range(self.problem_size):      # j-th dimension
                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = 2 * pi * uniform()
                    r3 = 2 * uniform()
                    r4 = uniform()
                    # Eq. 3.3, 3.1 and 3.2
                    if r4 < 0.5:
                        temp[j] = temp[j] + r1*sin(r2)*abs(r3*g_best[self.ID_POS][j] - temp[j])
                    else:
                        temp[j] = temp[j] + r1 * cos(r2) * abs( r3 * g_best[self.ID_POS][j] - temp[j])
                    # Check the bound
                    if temp[j] < self.domain_range[0] or temp[j] > self.domain_range[1]:
                        temp[j] = uniform(self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:               # My improved part
                    pop[i] = [temp, fit]

            ## Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalSCA(BaseSCA):
    """
    A Sine Cosine Algorithm for solving optimization problems (SCA)
    Link:
        https://doi.org/10.1016/j.knosys.2015.12.022
        https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
    """
    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        BaseSCA.__init__(self, root_paras, epoch, pop_size)

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Eq 3.4, r1 decreases linearly from a to 0
            a = 2.0
            r1 = a - (epoch+1) * (a / self.epoch)

            # Update the position of solutions with respect to destination
            for i in range(self.pop_size):              # i-th solution
                temp = deepcopy(pop[i][self.ID_POS])
                for j in range(self.problem_size):      # j-th dimension
                    # Update r2, r3, and r4 for Eq. (3.3)
                    r2 = 2 * pi * uniform()
                    r3 = 2 * uniform()
                    r4 = uniform()
                    # Eq. 3.3, 3.1 and 3.2
                    if r4 < 0.5:
                        temp[j] = pop[i][self.ID_POS][j] + r1*sin(r2)*abs(r3*g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                    else:
                        temp[j] = pop[i][self.ID_POS][j] + r1 * cos(r2) * abs( r3 * g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                    # Check the bound
                    if temp[j] < self.domain_range[0] or temp[j] > self.domain_range[1]:
                        temp[j] = uniform(self.domain_range[0], self.domain_range[1])

            # Re-calculate fitness
            for i in range(self.pop_size):
                pop[i][self.ID_FIT] = self._fitness_model__(pop[i][self.ID_FIT])

            ## Update the global best
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
