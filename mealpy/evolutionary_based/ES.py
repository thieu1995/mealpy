#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:14, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseES(Optimizer):
    """
        The original version of: Evolution Strategies (ES)
            (Clever Algorithms: Nature-Inspired Programming Recipes - Evolution Strategies)
    Link:
        http://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2      # strategy

    def __init__(self, problem: dict, epoch=1000, pop_size=100, n_child=0.75):
        """
        pop_size = miu, n_child = lamda
        Args:
            problem (dict): a dictionary of your problem
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            n_child (float/int): if float number --> percentage of child agents, int --> number of child agents
        """
        super().__init__(problem)
        self.epoch = epoch
        self.pop_size = pop_size        # miu
        if n_child < 1:                 # lamda, 75% of pop_size
            self.n_child = int(n_child * self.pop_size)
        else:
            self.n_child = int(n_child)
        self.distance = 0.05 * (self.ub - self.lb)

    def create_solution(self, minmax=0):
        pos = np.random.uniform(self.lb, self.ub)
        fit = self.get_fitness_position(pos)
        strategy = np.random.uniform(0, self.distance)
        return [pos, fit, strategy]

    def _mutate_solution_(self, solution=None):
        pos_new = solution[self.ID_POS] + solution[self.ID_STR] * np.random.normal(0, 1.0, self.problem_size)
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        tau = np.sqrt(2.0 * self.problem_size) ** -1.0
        tau_p = np.sqrt(2.0 * np.sqrt(self.problem_size)) ** -1.0
        strategy = np.exp(tau_p * np.random.normal(0, 1.0, self.problem_size) + tau * np.random.normal(0, 1.0, self.problem_size))
        return [pos_new, fit_new, strategy]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history_list_g_best = [g_best]
        self.history_list_c_best = self.history_list_g_best.copy()

        for epoch in range(0, self.epoch):
            time_start = time.time()

            children = [self._mutate_solution_(pop[i]) for i in range(0, self.n_child)]
            pop = pop + children
            pop = self.update_global_best_solution(pop)
            pop = pop[:self.pop_size]

            ## Additional information for the framework
            time_start = time.time() - time_start
            self.history_list_epoch_time.append(time_start)
            self.print_epoch(epoch + 1, time_start)
            self.history_list_pop.append(pop.copy())

        ## Additional information for the framework
        self.solution = self.history_list_g_best[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]


class LevyES(BaseES):
    """
        The levy version of: Evolution Strategies (ES)
        Noted:
            + Applied levy-flight
            + Change the flow of algorithm
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2  # strategy

    def __init__(self, problem: dict, epoch=750, pop_size=100, n_child=0.75):
        BaseES.__init__(self, problem, epoch, pop_size, n_child)

    def __create_levy_population__(self, pop=None):
        children = []
        for agent in pop:
            levy = self.get_step_size_levy_flight(multiplier=0.01, case=-1)
            pos_new = np.random.uniform(self.lb, self.ub) * levy * (agent[self.ID_POS] - self.history_list_g_best[-1][self.ID_POS])
            fit_new = self.get_fitness_position(pos_new)
            tau = np.sqrt(2.0 * self.problem_size) ** -1.0
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem_size)) ** -1.0
            stdevs = np.array([np.exp(tau_p * np.random.normal(0, 1.0) + tau * np.random.normal(0, 1.0)) for _ in range(self.problem_size)])
            children.append([pos_new, fit_new, stdevs])
        return children

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history_list_g_best = [g_best]
        self.history_list_c_best = self.history_list_g_best.copy()

        for epoch in range(0, self.epoch):
            time_start = time.time()

            children = [self._mutate_solution_(pop[i]) for i in range(0, self.n_child)]
            children_levy = self.__create_levy_population__(pop[self.n_child:])
            pop = pop + children + children_levy
            pop = self.update_global_best_solution(pop)
            pop = pop[:self.pop_size]

            ## Additional information for the framework
            time_start = time.time() - time_start
            self.history_list_epoch_time.append(time_start)
            self.print_epoch(epoch + 1, time_start)
            self.history_list_pop.append(pop.copy())

            ## Additional information for the framework
        self.solution = self.history_list_g_best[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

