#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import clip, sum, abs, cumsum, where, array
from numpy.random import uniform, normal
from copy import deepcopy
from mealpy.root import Root


class BaseSBO(Root):
    """
    My version of: Satin Bowerbird Optimizer (SBO)
        A new optimization algorithm to optimize ANFIS for software development effort estimation
    Notes:
        + Remove all third loop, n-times faster than original
        + No need equation (1, 2) in the paper, calculate probability by roulette-wheel. Also can handle negative values
        + Apply batch-size idea
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 alpha=0.94, pm=0.05, z=0.02, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha          # the greatest step size
        self.p_m = pm               # mutation probability
        self.z = z                  # percent of the difference between the upper and lower limit (Eq. 7)
        self.sigma = self.z * (self.ub - self.lb)   # proportion of space width

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            ## Calculate the probability of bowers using my equation
            fit_list = array([item[self.ID_FIT] for item in pop])

            for i in range(0, self.pop_size):
                ### Select a bower using roulette wheel
                idx = self.get_index_roulette_wheel_selection(fit_list)
                ### Calculating Step Size
                lamda = self.alpha * uniform()
                pos_new = pop[i][self.ID_POS] + lamda * ((pop[idx][self.ID_POS] + g_best[self.ID_POS]) / 2 - pop[i][self.ID_POS])
                ### Mutation
                temp = pop[i][self.ID_POS] + normal(0, 1, self.problem_size) * self.sigma
                pos_new = where(uniform(0, 1, self.problem_size) < self.p_m, temp, pos_new)
                ### In-bound position
                pos_new = clip(pos_new, self.lb, self.ub)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]

                ### Batch-size idea
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


class OriginalSBO(Root):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)
        A new optimization algorithm to optimize ANFIS for software development effort estimation
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017
        http://dx.doi.org/10.1016/j.engappai.2017.01.006
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 alpha=0.94, pm=0.05, z=0.02, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha              # the greatest step size
        self.p_m = pm                   # mutation probability
        self.z = z                      # percent of the difference between the upper and lower limit (Eq. 7)
        self.sigma = self.z * (self.ub - self.lb)     # proportion of space width

    def _roulette_wheel_selection__(self, fitness_list=None):
        r = uniform()
        c = cumsum(fitness_list)
        f = where(r < c)[0][0]
        return f

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

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
                        temp[j] = pop[i][self.ID_POS][j] + normal(0, 1) * self.sigma[j]
                temp = clip(temp, self.lb, self.ub)
                fit = self.get_fitness_position(temp)
                pop[i] = [temp, fit]

            ## Update elite if a bower becomes fitter than the elite
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
