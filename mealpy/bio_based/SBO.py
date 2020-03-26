#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import clip, sum, abs, cumsum, where, array
from numpy.random import uniform, normal
from copy import deepcopy
from mealpy.root import Root


class BaseSBO(Root):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)
    Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017
        http://dx.doi.org/10.1016/j.engappai.2017.01.006
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True, epoch=750, pop_size=100, alpha=0.94, pm=0.05, z=0.02):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha              # the greatest step size
        self.p_m = pm                   # mutation probability
        self.z = z                      # percent of the difference between the upper and lower limit (Eq. 7)
        self.sigma = self.z * (self.domain_range[1] - self.domain_range[0])     # proportion of space width

    def _roulette_wheel_selection__(self, fitness_list=None):
        r = uniform()
        c = cumsum(fitness_list)
        f = where(r < c)[0][0]
        return f

    def _train__(self):
        pop = [self._create_solution__() for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Calculate the probability of bowers using Eqs. (1) and (2)
            fx_list = array([item[self.ID_FIT] for item in pop])
            fit_list = deepcopy(fx_list)
            for i in range(0, self.pop_size):
                if fx_list[i] < 0:
                    fit_list[i] = 1.0 + abs(fx_list[i])
                else:
                    fit_list[i] = 1.0 / (1.0 + abs(fx_list[i]))
            fit_sum = sum(fit_list)
            ## Calculating the probability of each bower
            prob_list = fit_list / fit_sum

            for i in range(0, self.pop_size):
                temp = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    ### Select a bower using roulette wheel
                    idx = self._roulette_wheel_selection__(prob_list)
                    ### Calculating Step Size
                    lamda = self.alpha / (1 + prob_list[idx])
                    temp[j] = pop[i][self.ID_POS][j] + lamda * ( (pop[idx][self.ID_POS][j] + g_best[self.ID_POS][j]) / 2 - pop[i][self.ID_POS][j])
                    ### Mutation
                    if uniform() < self.p_m:
                        temp[j] = pop[i][self.ID_POS][j] + normal(0, 1) * self.sigma
                temp = clip(temp, self.domain_range[0], self.domain_range[1])
                fit = self._fitness_model__(temp)
                pop[i] = [temp, fit]

            ## Update elite if a bower becomes fitter than the elite
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
