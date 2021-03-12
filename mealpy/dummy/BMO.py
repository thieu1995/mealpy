#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:52, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal
from copy import deepcopy
from mealpy.root import Root


class OriginalBMO(Root):
    """
        The original version of: Blue Monkey Optimization
            (The blue monkey: A new nature inspired metaheuristic optimization algorithm)
        Link:
            http://dx.doi.org/10.21533/pen.v7i3.621
        Notes:
            + This is dummy algorithm and dummy paper.
            + The idea look like "Chicken Swarm Optimization"
            + The pseudo-code totally bullshit in my opinion, just read the paper you will understand.
            + The unclear point here is the "Rate equation": really confuse because It's contain the position. As you know,
            The position is the vector, but finally, the author conclude that Rate is random number in range [0, 1]
            + Luckily, using number we can plus/add number and vector or vector and vector.
            So at first, Rate is random number then after the 1st loop, its become vector.
            + Morever, both equtions movement of blue monkey and children is the same.
            + In addition, they don't check the bound after update position.
            + Keep going, they don't tell you the how to find the global best (I mean from blue monkey group or child group)
    """
    ID_POS = 0      # position
    ID_FIT = 1      # fitness
    ID_RAT = 2      # rate
    ID_WEI = 3      # weight

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, bm_teams=5, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.bm_teams = bm_teams                                # Number of blue monkey teams (5, 10, 20, ...)
        self.bm_size = int(self.pop_size/2)                     # Number of all blue monkey
        self.bm_numbers = int(self.bm_size / self.bm_teams)     # Number of blue monkey in each team

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position, minmax)
        rate = uniform(0, 1)
        weight = uniform(4, 6)
        return [position, fitness, rate, weight]

    def _create_population__(self):
        t1 = []
        for i in range(self.bm_size):
            t2 = [self.create_solution() for _ in range(self.bm_numbers)]
            t1.append(t2)
        t2 = [self.create_solution() for _ in range(self.bm_size)]
        return t1, t2

    def train(self):
        bm_pop, child_pop = self._create_population__()

        best = []
        for items in bm_pop:
            bt = self.get_global_best_solution(items, self.ID_FIT, self.ID_MIN_PROB)
            best.append(deepcopy(bt))
        g_best = deepcopy(self.get_global_best_solution(best, self.ID_FIT, self.ID_MIN_PROB))
        child_best = self.get_global_best_solution(child_pop, self.ID_FIT, self.ID_MIN_PROB)
        if g_best[self.ID_FIT] > child_best[self.ID_FIT]:
            g_best = deepcopy(child_best)

        for epoch in range(self.epoch):

            for id_bm, items in enumerate(bm_pop):
                items = sorted(items, key=lambda temp: temp[self.ID_FIT])
                if child_best[self.ID_FIT] < items[-1][self.ID_FIT]:
                    bm_pop[id_bm][-1] = deepcopy(child_best)

            for id_bm, items in enumerate(bm_pop):
                leader = self.get_global_best_solution(items, self.ID_FIT, self.ID_MIN_PROB)
                for i in range(self.bm_numbers):
                    rate = 0.7 * items[i][self.ID_RAT] + (leader[self.ID_WEI] - items[i][self.ID_WEI]) * \
                              uniform() * (leader[self.ID_POS] - items[i][self.ID_POS])
                    pos = items[i][self.ID_POS] + uniform() * rate
                    pos = self.amend_position_faster(pos)
                    fit = self.get_fitness_position(pos)
                    if fit < items[i][self.ID_FIT]:
                        we = items[i][self.ID_WEI]
                        bm_pop[id_bm][i] = [pos, fit, rate, we]

            for i in range(self.bm_size):
                rate = 0.7 * child_pop[i][self.ID_RAT] + (child_best[self.ID_WEI] - child_pop[i][self.ID_WEI]) * \
                    uniform() * (child_best[self.ID_POS] - child_pop[i][self.ID_POS])
                pos = child_pop[i][self.ID_POS] + uniform() * rate
                pos = self.amend_position_faster(pos)
                fit = self.get_fitness_position(pos)
                if fit < child_pop[i][self.ID_FIT]:
                    we = child_pop[i][self.ID_WEI]
                    child_pop[i] = [pos, fit, rate, we]

            child_best = self.update_global_best_solution(child_pop, self.ID_MIN_PROB, child_best)
            if child_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(child_best)

            for i in range(self.bm_teams):
                bt = self.get_global_best_solution(bm_pop[i], self.ID_FIT, self.ID_MIN_PROB)
                best[i] = bt
            g_best = self.update_global_best_solution(best, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseBMO(Root):
    """
        My modified version of: Blue Monkey Optimization
    """

    ID_RATE = 2

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, bm_teams=5, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.bm_teams = bm_teams                                # Number of blue monkey teams (5, 10, 20, ...)
        self.bm_size = int(self.pop_size/2)                     # Number of all blue monkey
        self.bm_numbers = int(self.bm_size / self.bm_teams)     # Number of blue monkey in each team
        self.w = [4, 6]

    def create_solution(self, minmax=0):
        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position, minmax)
        rate = uniform(0, 1, self.problem_size)
        return [position, fitness, rate]

    def _create_population__(self):
        t1 = []
        for i in range(self.bm_size):
            t1.append([self.create_solution() for _ in range(self.bm_numbers)])
        t2 = [self.create_solution() for _ in range(self.bm_size)]
        return t1, t2

    def train(self):
        bm_pop, child_pop = self._create_population__()

        best = []
        for items in bm_pop:
            best.append(self.get_global_best_solution(items, self.ID_FIT, self.ID_MIN_PROB))
        g_best = self.get_global_best_solution(best, self.ID_FIT, self.ID_MIN_PROB)
        child_best = self.get_global_best_solution(child_pop, self.ID_FIT, self.ID_MIN_PROB)
        if g_best[self.ID_FIT] > child_best[self.ID_FIT]:
            g_best = deepcopy(child_best)

        for epoch in range(self.epoch):
            w = (self.epoch - epoch) / self.epoch * (self.w[1] - self.w[0]) + self.w[0]
            child_pop = sorted(child_pop, key=lambda temp: temp[self.ID_FIT])
            for i in range(self.bm_teams):
                bm_pop[i] = sorted(bm_pop[i], key=lambda temp: temp[self.ID_FIT])
                if child_pop[i][self.ID_FIT] < bm_pop[i][-1][self.ID_FIT]:
                    bm_pop[i][-1] = deepcopy(child_pop[i])
                    child_pop[i] = self.create_solution()

            for id_bm, items in enumerate(bm_pop):
                leader = self.get_global_best_solution(items, self.ID_FIT, self.ID_MIN_PROB)
                for i in range(self.bm_numbers):
                    rate = items[i][self.ID_RATE] + w * uniform(0, 1, self.problem_size) * (leader[self.ID_POS] - items[i][self.ID_POS])
                    if uniform() < 0.5:
                        pos_new = items[i][self.ID_POS] + uniform() * rate
                    else:
                        pos_new = self.levy_flight(epoch, items[i][self.ID_POS], g_best[self.ID_POS])
                    pos_new = self.amend_position_faster(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    if fit_new < items[i][self.ID_FIT]:
                        items[i] = [pos_new, fit_new, rate]

            for i in range(self.bm_size):
                rate = child_pop[i][self.ID_RATE] + w * normal(0, 1, self.problem_size) * (g_best[self.ID_POS] - child_pop[i][self.ID_POS])
                pos = child_pop[i][self.ID_POS] + normal() * rate
                pos = self.amend_position_faster(pos)
                fit = self.get_fitness_position(pos)
                if fit < child_pop[i][self.ID_FIT]:
                    child_pop[i] = [pos, fit, rate]

            child_best = self.update_global_best_solution(child_pop, self.ID_MIN_PROB, child_best)
            if child_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(child_best)

            for i in range(self.bm_teams):
                bt = self.get_global_best_solution(bm_pop[i], self.ID_FIT, self.ID_MIN_PROB)
                best[i] = bt
            g_best = self.update_global_best_solution(best, self.ID_MIN_PROB, g_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
