#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import zeros
from copy import deepcopy
from mealpy.root import Root


class BaseBOA(Root):
    """
            Butterfly Optimization Algorithm (BOA)
        This is the version I implemented as the paper:
        Butterfly optimization algorithm: a novel approach for global optimization
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c=0.01, p=0.8, alpha=(0.1, 0.3), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c = c              # 0.01, is the sensory modality
        self.p = p              # 0.8, Search for food and mating partner by butterflies can occur at both local and global scale
        self.alpha = alpha      # 0.1-0.3 (0 -> finite), the power exponent dependent on modality

    def train(self):
        alpha = self.alpha[0]
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        list_fragrance = zeros(self.pop_size)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):
                list_fragrance[i] = self.c * pop[i][self.ID_FIT]**alpha

            for i in range(self.pop_size):
                r = uniform()
                if r < self.p:
                    pos_new = pop[i][self.ID_POS] + (r**2 * g_best[self.ID_POS] - pop[i][self.ID_POS]) * list_fragrance[i]
                else:
                    idx = randint(0, self.pop_size)
                    pos_new = pop[i][self.ID_POS] + (r**2 * pop[idx][self.ID_POS] - pop[i][self.ID_POS]) * list_fragrance[i]
                fit_new = self.get_fitness_position(pos_new)
                fra = self.c * fit_new**alpha
                pop[i] = [pos_new, fit_new, fra]
            alpha = self.alpha[0] + ((epoch + 1)/self.epoch) * (self.alpha[1] - self.alpha[0])

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalBOA(Root):
    """
        The original version of: Butterfly Optimization Algorithm (BOA)
            (Butterfly optimization algorithm: a novel approach for global optimization)
        Notes:
            + This algorithm and paper is dummy.
            + This is the code of the original author of BOA. He public on mathworks. But take a look at his code and his paper. That is completely different.
            + I implement this version based on his paper, it can't converge at all.
        https://www.mathworks.com/matlabcentral/fileexchange/68209-butterfly-optimization-algorithm-boa

            + So many people asking him public the code of function, which used in the paper. Even 1 guy said
        "Honestly,this algorithm looks like Flower Pollination Algorithm developed by Yang."
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c=0.01, p=0.8, alpha=(0.1, 0.3), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c = c                  # 0.01, is the sensory modality
        self.p = p                  # 0.8, Search for food and mating partner by butterflies can occur at both local and global scale
        self.alpha = alpha          # 0.1-0.3 (0 -> finite), the power exponent dependent on modality

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        c_temp = self.c
        alpha = self.alpha[0]

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                FP = c_temp * (pop[i][self.ID_FIT] ** alpha)

                if uniform() < self.p:
                    t1 = pop[i][self.ID_POS] + (uniform()*uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]) * FP
                else:
                    epsilon = uniform()
                    id1, id2 = randint(0, self.pop_size, 2)
                    dis = (epsilon**2) * pop[id1][self.ID_POS] - pop[id2][self.ID_POS]
                    t1 = pop[i][self.ID_POS] + dis * FP
                t1 = self.amend_position_faster(t1)
                fit = self.get_fitness_position(t1)

                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [deepcopy(t1), fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [deepcopy(t1), fit]

            c_temp = c_temp + 0.025 / (c_temp * self.epoch)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class AdaptiveBOA(OriginalBOA):
    """
    The adaptive version of: Butterfly Optimization Algorithm (BOA)
    Links:
        + A novel adaptive butterfly optimization algorithm

    Wow, my mind just blown up when I found out that this guy:
        https://scholar.google.co.in/citations?hl=en&user=KvcHovcAAAAJ&view_op=list_works&sortby=pubdate
    He invent BOA algorithm and public it in 2019, but so many variant version of BOA has been created since 2015.
    How the hell that happened?
    This is a plagiarism? I think this is one of the most biggest reason why mathematician researchers calling out
        meta-heuristics community is completely bullshit and unethical.
    Just for producing more trash paper without any knowledge in it? This is why I listed BOA as the totally trash and dummy
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c=0.01, p=0.8, alpha=(0.1, 0.3), **kwargs):
        OriginalBOA.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, c, p, alpha, kwargs = kwargs)


    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)
        c_temp = self.c
        alpha = self.alpha[0]

        for epoch in range(self.epoch):

            for i in range(self.pop_size):
                FP = c_temp * (pop[i][self.ID_FIT] ** alpha)

                if uniform() < self.p:
                    t1 = pop[i][self.ID_POS] + (uniform()*uniform() * g_best[self.ID_POS] - pop[i][self.ID_POS]) * FP
                else:
                    epsilon = uniform()
                    id1, id2 = randint(0, self.pop_size, 2)
                    dis = (epsilon**2) * pop[id1][self.ID_POS] - pop[id2][self.ID_POS]
                    t1 = pop[i][self.ID_POS] + dis * FP
                t1 = self.amend_position_faster(t1)
                fit = self.get_fitness_position(t1)

                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [deepcopy(t1), fit]

                if fit < g_best[self.ID_FIT]:
                    g_best = [deepcopy(t1), fit]

            c_temp = c_temp * (10.0**(-5)/0.9)**(2.0 / self.epoch)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train



