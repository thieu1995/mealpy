#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 07:03, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import exp, sign, ones
from numpy.random import uniform, randint
from copy import deepcopy
from mealpy.root import Root


class BaseEO(Root):
    """
    Original: Equilibrium Optimizer (EO)
        (Equilibrium Optimizer: A Novel Optimization Algorithm)
        https://doi.org/10.1016/j.knosys.2019.105190
        https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        V = 1
        a1 = 2
        a2 = 1
        GP = 0.5
        #c_eq1 = [None, float("inf")]                    # it is global best solution
        c_eq2 = [None, float("inf")]
        c_eq3 = [None, float("inf")]
        c_eq4 = [None, float("inf")]

        # ---------------- Memory saving-------------------
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        c_eq1 = deepcopy(g_best)

        for epoch in range(0, self.epoch):

            for i in range(0, self.pop_size):

                if pop[i][self.ID_FIT] < c_eq1[self.ID_FIT]:
                    c_eq1 = deepcopy(pop[i])
                elif c_eq1[self.ID_FIT] < pop[i][self.ID_FIT] and pop[i][self.ID_FIT] < c_eq2[self.ID_FIT]:
                    c_eq2 = deepcopy(pop[i])
                elif c_eq1[self.ID_FIT] < pop[i][self.ID_FIT] and c_eq2[self.ID_FIT] < pop[i][self.ID_FIT] and pop[i][self.ID_FIT] < c_eq3[self.ID_FIT]:
                    c_eq3 = deepcopy(pop[i])
                elif c_eq1[self.ID_FIT] < pop[i][self.ID_FIT] and c_eq2[self.ID_FIT] < pop[i][self.ID_FIT] and c_eq3[self.ID_FIT] < pop[i][self.ID_FIT] and pop[i][self.ID_FIT] < c_eq4[self.ID_FIT]:
                    c_eq4 = deepcopy(pop[i])

            # make equilibrium pool
            c_eq_ave = (c_eq1[self.ID_POS] + c_eq2[self.ID_POS] + c_eq3[self.ID_POS] + c_eq4[self.ID_POS]) / 4
            fit_ave = self._fitness_model__(c_eq_ave)
            c_pool = [c_eq1, c_eq2, c_eq3, c_eq4, [c_eq_ave, fit_ave]]

            # Eq. 9
            t = (1 - epoch/self.epoch) ** (a2 * epoch / self.epoch)

            for i in range(0, self.pop_size):
                lamda = uniform(0, 1, self.problem_size)                # lambda in Eq. 11
                r = uniform(0, 1, self.problem_size)                    # r in Eq. 11
                c_eq = c_pool[randint(0, len(c_pool))][self.ID_POS]     # random selection 1 of candidate from the pool
                f = a1 * sign(r - 0.5) * (exp(-lamda * t) - 1.0)        # Eq. 11
                r1 = uniform()
                r2 = uniform()                                                                  # r1, r2 in Eq. 15
                gcp = 0.5 * r1 * ones(self.problem_size) * (r2 >= GP)                           # Eq. 15
                g0 = gcp * (c_eq - lamda * pop[i][self.ID_POS])                                 # Eq. 14
                g = g0 * f                                                                      # Eq. 13
                temp = c_eq + (pop[i][self.ID_POS] - c_eq) * f + (g * V / lamda) * (1.0 - f)    # Eq. 16
                fit = self._fitness_model__(temp)
                pop[i] = [temp, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
