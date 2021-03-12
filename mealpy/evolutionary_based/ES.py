#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:14, 10/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
from numpy import sqrt, exp, array
from mealpy.root import Root


class BaseES(Root):
    """
        The original version of: Evolution Strategies (ES)
            (Clever Algorithms: Nature-Inspired Programming Recipes - Evolution Strategies)
    Link:
        http://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_STR = 2      # strategy

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_child=0.75, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size        # miu
        if n_child < 1:                 # lamda, 75% of pop_size
            self.n_child = int(n_child * self.pop_size)
        else:
            self.n_child = int(n_child)
        self.distance = 0.05 * (self.ub - self.lb)

    def create_solution(self, minmax=0):
        pos = uniform(self.lb, self.ub)
        fit = self.get_fitness_position(pos)
        strategy = uniform(0, self.distance, self.problem_size)
        return [pos, fit, strategy]

    def _mutate_solution__(self, solution=None):
        child = solution[self.ID_POS] + solution[self.ID_STR] * normal(0, 1.0, self.problem_size)
        child = self.amend_position_faster(child)
        fit = self.get_fitness_position(child)
        tau = sqrt(2.0 * self.problem_size) ** -1.0
        tau_p = sqrt(2.0 * sqrt(self.problem_size)) ** -1.0
        strategy = exp(tau_p * normal(0, 1.0, self.problem_size) + tau * normal(0, 1.0, self.problem_size))
        return [child, fit, strategy]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            children = [self._mutate_solution__(pop[i]) for i in range(0, self.n_child)]
            pop = pop + children
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            # Update the global best
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


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

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, n_child=0.75, **kwargs):
        BaseES.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, n_child, kwargs=kwargs)

    def __create_levy_population__(self, epoch=None, g_best=None, pop=None):
        children = []
        for sol in pop:
            pos = self.levy_flight(epoch, sol[self.ID_POS], g_best[self.ID_POS])
            fit = self.get_fitness_position(pos)
            tau = sqrt(2.0 * self.problem_size) ** -1.0
            tau_p = sqrt(2.0 * sqrt(self.problem_size)) ** -1.0
            stdevs = array([exp(tau_p * normal(0, 1.0) + tau * normal(0, 1.0)) for _ in range(self.problem_size)])
            children.append([pos, fit, stdevs])
        return children

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            children = [self._mutate_solution__(pop[i]) for i in range(0, self.n_child)]
            children_levy = self.__create_levy_population__(epoch, g_best, pop[self.n_child:])
            pop = pop + children + children_levy
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            pop = pop[:self.pop_size]
            # Update the global best
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

