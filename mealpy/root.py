#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:58, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, sqrt, sign, ones, ptp, min, sum
from numpy.random import uniform, random, normal
from math import gamma
from copy import deepcopy


class Root:
    """ This is root of all Algorithms """

    ID_MIN_PROB = 0  # min problem
    ID_MAX_PROB = -1  # max problem

    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-10

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True):
        """
        Parameters
        ----------
        objective_func :
        problem_size :
        domain_range :
        log :
        """
        self.objective_func = objective_func
        self.problem_size = problem_size
        self.domain_range = domain_range
        self.log = log
        self.solution, self.loss_train = None, []

    def _create_solution__(self, minmax=0):
        """
        Return the encoded solution with 2 element: position of solution and fitness of solution

        Parameters
        ----------
        minmax
            0 - minimum problem, else - maximum problem

        """
        solution = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(solution=solution, minmax=minmax)
        return [solution, fitness]

    def _fitness_model__(self, solution=None, minmax=0):
        """     Assumption that objective function always return the original value
        :param solution: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
        return self.objective_func(solution) if minmax == 0 else 1.0 / (self.objective_func(solution) + self.EPSILON)

    def _fitness_encoded__(self, encoded=None, id_pos=None, minmax=0):
        return self._fitness_model__(solution=encoded[id_pos], minmax=minmax)

    def _get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return deepcopy(sorted_pop[id_best])

    def _sort_pop_and_get_global_best__(self, pop=None, id_fitness=None, id_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[id_fitness])
        return sorted_pop, deepcopy(sorted_pop[id_best])

    def _amend_solution__(self, solution=None):
        return maximum(self.domain_range[0], minimum(self.domain_range[1], solution))

    def _amend_solution_faster__(self, solution=None):
        return clip(solution, self.domain_range[0], self.domain_range[1])

    def _amend_solution_random__(self, solution=None):
        for i in range(self.problem_size):
            if solution[i] < self.domain_range[0] or solution[i] > self.domain_range[1]:
                solution[i] = uniform(self.domain_range[0], self.domain_range[1])
        return solution

    def _amend_solution_random_faster__(self, solution=None):
        return where(logical_and(self.domain_range[0] <= solution, solution <= self.domain_range[1]), solution, uniform(self.domain_range[0],
                                                                                                                        self.domain_range[1]))

    def _update_global_best__(self, pop=None, id_best=None, g_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        return deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)

    def _sort_pop_and_update_global_best__(self, pop=None, id_best=None, g_best=None):
        sorted_pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        current_best = sorted_pop[id_best]
        g_best = deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)
        return sorted_pop, g_best

    def _create_opposition_solution__(self, solution=None, g_best=None):
        t1 = self.domain_range[1] * ones(self.problem_size) + self.domain_range[0] * ones(self.problem_size)
        t2 = -1 * g_best[self.ID_POS] + uniform() * (g_best - solution)
        return t1 + t2

    def _levy_flight__(self, epoch=None, solution=None, g_best=None, step=0.001, case=0):
        """
        Parameters
        ----------
        epoch (int): current iteration
        solution : 1-D numpy array
        g_best : 1-D numpy array
        step (float): 0.001
        case (int): 0, 1, 2

        """
        beta = 1
        # muy and v are two random variables which follow normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = power(gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = normal(0, sigma_muy ** 2)
        v = normal(0, sigma_v ** 2)
        s = muy / power(abs(v), 1 / beta)
        # D is a random solution
        D = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        LB = step * s * (solution - g_best)
        levy = D * LB

        if case == 0:
            return levy
        elif case == 1:
            return solution + 1.0 / sqrt(epoch + 1) * sign(random() - 0.5) * levy
        elif case == 2:
            return solution + 0.01 * levy

    def _levy_flight_2__(self, solution=None, g_best=None):
        alpha = 0.01
        xichma_v = 1
        xichma_u = ((gamma(1 + 1.5) * sin(pi * 1.5 / 2)) / (gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (normal(0, xichma_u ** 2)) / (sqrt(abs(normal(0, xichma_v ** 2))) ** (1.0 / 1.5))
        return solution[self.ID_POS] + alpha * levy_b * (solution - g_best)

    def _get_index_roulette_wheel_selection_(self, list_fitness=None):
        """ It can handle negative also. Make sure your list fitness is 1D-numpy array"""
        scaled_fitness = (list_fitness - min(list_fitness)) / ptp(list_fitness)
        minimized_fitness = 1.0 - scaled_fitness
        total_sum = sum(minimized_fitness)
        r = uniform(low=0, high=total_sum)
        for idx, f in enumerate(minimized_fitness):
            r = r + f
            if r > total_sum:
                return idx

    def _train__(self):
        pass
