#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 14:01, 16/11/2020                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, abs, exp, cos, pi
from numpy.random import uniform, randint, rand
from numpy.linalg import norm
from mealpy.root import Root


class OriginalFOA(Root):
    """
        The original version of: Fruit-fly Optimization Algorithm (FOA)
            (A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example)
        Link:
            DOI: https://doi.org/10.1016/j.knosys.2011.07.001
        Notes:
            + This optimization can't apply to complicated objective function in this library.
            + So I changed the implementation Original FOA in BaseFOA version
            + This algorithm is the weakest algorithm in MHAs (not count fakes algorithms), that's why so many researchers produce paper based
            on this algorithm (Easy to improve, and easy to implement).
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        s = array([1.0 / norm(position)])
        fitness = self.get_fitness_position(position=s, minmax=minmax)
        return [position, fitness]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                pos_new = pop[i][self.ID_POS] + uniform(self.lb, self.ub)
                fit = self.get_fitness_position(pos_new)
                pop[i] = [pos_new, fit]
            ## Update the global best
            g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseFOA(Root):
    """
        My version of: Fruit-fly Optimization Algorithm (FOA)
            (A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example)
        Notes:
            + 1) I changed the fitness function (smell function) by taking the distance each 2 adjacent dimensions
            + 2) Update the position if only it find the better fitness value.
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def norm_consecutive_adjacent(self, position=None):
        return array([norm([position[x], position[x+1]]) for x in range(0, self.problem_size-1)] + [uniform()])

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        s = self.norm_consecutive_adjacent(position)
        fitness = self.get_fitness_position(position=s, minmax=minmax)      # Since the problem is minimize so no need 1.0/s
        return [position, fitness]

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(0, self.pop_size):
                pos_new = pop[i][self.ID_POS] + uniform(self.lb, self.ub)
                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]
                ## Update the global best based on batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class WFOA(BaseFOA):
    """
        The original version of: Whale Fruit-fly Optimization Algorithm (WFOA)
            (Boosted Hunting-based Fruit Fly Optimization and Advances in Real-world Problems)
        Link:
            https://doi.org/10.1016/j.eswa.2020.113502
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseFOA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            a = 2 - 2 * epoch / (self.epoch - 1)  # linearly decreased from 2 to 0
            for i in range(self.pop_size):
                r = rand()
                A = 2 * a * r - a
                C = 2 * r
                l = uniform(-1, 1)
                p = 0.5
                b = 1
                if uniform() < p:
                    if abs(A) < 1:
                        D = abs(C * g_best[self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = g_best[self.ID_POS] - A * D
                    else:
                        x_rand = pop[randint(self.pop_size)]         # select random 1 position in pop
                        D = abs(C * x_rand[self.ID_POS] - pop[i][self.ID_POS])
                        pos_new = (x_rand[self.ID_POS] - A * D)
                else:
                    D1 = abs(g_best[self.ID_POS] - pop[i][self.ID_POS])
                    pos_new = D1 * exp(b * l) * cos(2 * pi * l) + g_best[self.ID_POS]

                pos_new = self.amend_position_faster(pos_new)
                smell = self.norm_consecutive_adjacent(pos_new)
                fit = self.get_fitness_position(smell)
                pop[i] = [pos_new, fit]

                ## batch size idea
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