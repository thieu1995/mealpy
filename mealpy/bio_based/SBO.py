#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:48, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
from mealpy.optimizer import Optimizer


class BaseSBO(Optimizer):
    """
    My version of: Satin Bowerbird Optimizer (SBO)
        A new optimization algorithm to optimize ANFIS for software development effort estimation
    Link:
        https://doi.org/10.1016/j.engappai.2017.01.006
    Notes:
        + Remove all third loop, n-times faster than original
        + No need equation (1, 2) in the paper, calculate probability by roulette-wheel. Also can handle negative values
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.94, pm=0.05, psw=0.02, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            pm (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.p_m = pm
        self.psw = psw
        # (percent of the difference between the upper and lower limit (Eq. 7))
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        ## Calculate the probability of bowers using my equation
        fit_list = np.array([item[self.ID_FIT][self.ID_TAR] for item in pop])

        for i in range(0, self.pop_size):
            ### Select a bower using roulette wheel
            idx = self.get_index_roulette_wheel_selection(fit_list)
            ### Calculating Step Size
            lamda = self.alpha * np.random.uniform()
            pos_new = pop[i][self.ID_POS] + lamda * ((pop[idx][self.ID_POS] + g_best[self.ID_POS]) / 2 - pop[i][self.ID_POS])
            ### Mutation
            temp = pop[i][self.ID_POS] + np.random.normal(0, 1, self.problem.n_dims) * self.sigma
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, temp, pos_new)
            ### In-bound position
            pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            pop[i][self.ID_POS] = pos_new

        pop = self.update_fitness_population(mode, pop)
        return pop


class OriginalSBO(BaseSBO):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)
        A new optimization algorithm to optimize ANFIS for software development effort estimation
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017
        http://dx.doi.org/10.1016/j.engappai.2017.01.006
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.94, pm=0.05, psw=0.02, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            pm (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, alpha, pm, psw, **kwargs)

    def _roulette_wheel_selection__(self, fitness_list=None):
        r = np.random.uniform()
        c = np.cumsum(fitness_list)
        f = np.where(r < c)[0][0]
        return f

    def evolve(self, mode='sequential', epoch=None, pop=None, g_best=None):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        ## Calculate the probability of bowers using Eqs. (1) and (2)
        fx_list = np.array([agent[self.ID_FIT][self.ID_TAR] for agent in pop])
        fit_list = fx_list.copy()
        for i in range(0, self.pop_size):
            if fx_list[i] < 0:
                fit_list[i] = 1.0 + np.abs(fx_list[i])
            else:
                fit_list[i] = 1.0 / (1.0 + np.abs(fx_list[i]))
        fit_sum = np.sum(fit_list)
        ## Calculating the probability of each bower
        prob_list = fit_list / fit_sum

        for i in range(0, self.pop_size):
            pos_new = pop[i][self.ID_POS].copy()
            for j in range(0, self.problem.n_dims):
                ### Select a bower using roulette wheel
                idx = self._roulette_wheel_selection__(prob_list)
                ### Calculating Step Size
                lamda = self.alpha / (1 + prob_list[idx])
                pos_new[j] = pop[i][self.ID_POS][j] + lamda * ((pop[idx][self.ID_POS][j] + g_best[self.ID_POS][j]) / 2 - pop[i][self.ID_POS][j])
                ### Mutation
                if np.random.uniform() < self.p_m:
                    pos_new[j] = pop[i][self.ID_POS][j] + np.random.normal(0, 1) * self.sigma[j]
            pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            pop[i][self.ID_POS] = pos_new
        pop = self.update_fitness_population(mode, pop)
        return pop

