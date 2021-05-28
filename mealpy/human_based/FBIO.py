#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:57, 14/06/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import array, floor, min, max, where
from numpy.random import uniform, choice, normal, randint
from copy import deepcopy
from mealpy.root import Root


class BaseFBIO(Root):
    """
    My modified version of: Forensic-Based Investigation Optimization (FBIO)
        (FBI inspired meta-optimization)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S1568494620302799
    Notes:
        + Implement the fastest way (Remove all third loop)
        + Change equations
        + Change the flow of algorithm
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def probability(self, list_fitness=None):  # Eq.(3) in FBI Inspired Meta-Optimization
        max1 = max(list_fitness)
        min1 = min(list_fitness)
        prob = (max1 - list_fitness) / (max1 - min1 + self.EPSILON)
        return prob

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Optimization Cycle
        for epoch in range(self.epoch):
            # Investigation team - team A
            # Step A1
            for i in range(0, self.pop_size):
                n_change = randint(0, self.problem_size)
                nb1, nb2 = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)
                # Eq.(2) in FBI Inspired Meta - Optimization
                pos_a = deepcopy(pop[i][self.ID_POS])
                pos_a[n_change] = pop[i][self.ID_POS][n_change] + normal() * (pop[i][self.ID_POS][n_change] -
                                                                            (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change]) / 2)
                pos_a = self.amend_position_random_faster(pos_a)
                fit_a = self.get_fitness_position(pos_a)
                if fit_a < pop[i][self.ID_FIT]:
                    pop[i] = [pos_a, fit_a]
                    if fit_a < g_best[self.ID_FIT]:
                        g_best = [pos_a, fit_a]
            # Step A2
            list_fitness = array([item[self.ID_FIT] for item in pop])
            prob = self.probability(list_fitness)
            for i in range(0, self.pop_size):
                if uniform() > prob[i]:
                    r1, r2, r3 = choice(list(set(range(0, self.pop_size)) - {i}), 3, replace=False)
                    ## Remove third loop here, the condition also not good, need to remove also. No need Rnd variable
                    pos_a = deepcopy(pop[i][self.ID_POS])
                    temp = g_best[self.ID_POS] + pop[r1][self.ID_POS] + uniform() * (pop[r2][self.ID_POS] - pop[r3][self.ID_POS])
                    pos_a = where(uniform(0, 1, self.problem_size) < 0.5, temp, pos_a)
                    pos_a = self.amend_position_random_faster(pos_a)
                    fit_a = self.get_fitness_position(pos_a)
                    if fit_a < pop[i][self.ID_FIT]:
                        pop[i] = [pos_a, fit_a]
                        if fit_a < g_best[self.ID_FIT]:
                            g_best = [pos_a, fit_a]
            ## Persuing team - team B
            ## Step B1
            for i in range(0, self.pop_size):
                ### Remove third loop here also
                ### Eq.(6) in FBI Inspired Meta-Optimization
                pos_b = uniform(0, 1, self.problem_size) * pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ## Step B2
            for i in range(0, self.pop_size):
                rr = choice(list(set(range(0, self.pop_size)) - {i}))
                if pop[i][self.ID_FIT] > pop[rr][self.ID_FIT]:
                    ## Eq.(7) in FBI Inspired Meta-Optimization
                    pos_b = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[rr][self.ID_POS] - pop[i][self.ID_POS]) + \
                            uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
                else:
                    ## Eq.(8) in FBI Inspired Meta-Optimization
                    pos_b = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[i][self.ID_POS] - pop[rr][self.ID_POS]) + \
                            uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalFBIO(BaseFBIO):
    """
    The original version of: Forensic-Based Investigation Optimization (FBIO)
        (FBI inspired meta-optimization)
    Link:
        DOI: https://doi.org/10.1016/j.asoc.2020.106339
        Matlab code: https://ww2.mathworks.cn/matlabcentral/fileexchange/76299-forensic-based-investigation-algorithm-fbi
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseFBIO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]

        # Memorize the best solution
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # Optimization Cycle
        for epoch in range(self.epoch):
            # Investigation team - team A
            # Step A1
            for i in range(0, self.pop_size):
                n_change = randint(0, self.problem_size)
                nb1 = randint(0, self.pop_size)
                ## Not good move here, using 2 while loop to select different random solution
                while (nb1 == i):
                    nb1 = randint(0, self.pop_size)
                nb2 = randint(0, self.pop_size)
                while (nb1 == nb2 or nb2 == i):
                    nb2 = randint(0, self.pop_size)

                # Eq.(2) in FBI Inspired Meta - Optimization
                pos_a = deepcopy(pop[i][self.ID_POS])
                pos_a[n_change] = pop[i][self.ID_POS][n_change] + (uniform() - 0.5)*2 * (pop[i][self.ID_POS][n_change] -
                                                                               (pop[nb1][self.ID_POS][n_change] + pop[nb2][self.ID_POS][n_change])/2)
                ## Not good move here, change only 1 variable but check bound of all variable in solution
                pos_a = self.amend_position_random_faster(pos_a)
                fit_a = self.get_fitness_position(pos_a)
                if fit_a < pop[i][self.ID_FIT]:
                    pop[i] = [pos_a, fit_a]
                    if fit_a < g_best[self.ID_FIT]:
                        g_best = [pos_a, fit_a]
            # Step A2
            list_fitness = array([item[self.ID_FIT] for item in pop])
            prob = self.probability(list_fitness)
            for i in range(0, self.pop_size):
                if uniform() > prob[i]:
                    ## Same above, not good move here, using 3 while loop to select 3 different random solution
                    r1 = randint(0, self.pop_size)
                    while r1 == i:
                        r1 = randint(0, self.pop_size)
                    r2 = randint(0, self.pop_size)
                    while (r2 == r1) or (r2 == i):
                        r2 = randint(0, self.pop_size)
                    r3 = randint(0, self.pop_size)
                    while (r3 == r2) or (r3 == r1) or (r3 == i):
                        r3 = randint(0, self.pop_size)
                    pos_a = deepcopy(pop[i][self.ID_POS])
                    Rnd = floor(uniform() * self.problem_size) + 1

                    for j in range(0, self.problem_size):
                        if (uniform() < uniform() or Rnd == j):
                            pos_a[j] = g_best[self.ID_POS][j] + pop[r1][self.ID_POS][j] + uniform() * (pop[r2][self.ID_POS][j] - pop[r3][self.ID_POS][j])
                        ## In the original matlab code they do the else condition here, not good again because no need else here
                    ## Same here, they do check the bound of all variable in solution
                    pos_a = self.amend_position_random_faster(pos_a)
                    fit_a = self.get_fitness_position(pos_a)
                    if fit_a < pop[i][self.ID_FIT]:
                        pop[i] = [pos_a, fit_a]
                        if fit_a < g_best[self.ID_FIT]:
                            g_best = [pos_a, fit_a]
            ## Persuing team - team B
            ## Step B1
            for i in range(0, self.pop_size):
                pos_b = deepcopy(pop[i][self.ID_POS])
                for j in range(0, self.problem_size):
                    ### Eq.(6) in FBI Inspired Meta-Optimization
                    pos_b[j] = uniform() * pop[i][self.ID_POS][j] + uniform() * (g_best[self.ID_POS][j] - pop[i][self.ID_POS][j])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            ## Step B2
            for i in range(0, self.pop_size):
                ### Not good move here again
                rr = randint(0, self.pop_size)
                while rr == i:
                    rr = randint(0, self.pop_size)
                if pop[i][self.ID_FIT] > pop[rr][self.ID_FIT]:
                    ## Eq.(7) in FBI Inspired Meta-Optimization
                    pos_b = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[rr][self.ID_POS] - pop[i][self.ID_POS]) + \
                            uniform() * (g_best[self.ID_POS] - pop[rr][self.ID_POS])
                else:
                    ## Eq.(8) in FBI Inspired Meta-Optimization
                    pos_b = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (pop[i][self.ID_POS] - pop[rr][self.ID_POS]) + \
                            uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                pos_b = self.amend_position_random_faster(pos_b)
                fit_b = self.get_fitness_position(pos_b)
                if fit_b < pop[i][self.ID_FIT]:
                    pop[i] = [pos_b, fit_b]
                    if fit_b < g_best[self.ID_FIT]:
                        g_best = [pos_b, fit_b]

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
