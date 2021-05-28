#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:14, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, choice, rand
from numpy import array, mean, setxor1d
from copy import deepcopy
from mealpy.root import Root


class BaseTLO(Root):
    """
        Teaching-Learning-based Optimization (TLO)
    An elitist teaching-learning-based optimization algorithm for solving complex constrained optimization problems(TLO)
        This is my version taken the advantages of numpy array to faster handler operations.
    Notes:
        + Remove all third loop
        + Using global best solution
        + Using batch-size idea
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Teaching Phrase
                TF = randint(1, 3)  # 1 or 2 (never 3)
                list_pos = array([item[self.ID_POS] for item in pop])
                DIFF_MEAN = rand(self.problem_size) * (g_best[self.ID_POS] - TF * mean(list_pos, axis=0))
                temp = pop[i][self.ID_POS] + DIFF_MEAN
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

                ## Learning Phrase
                temp = deepcopy(pop[i][self.ID_POS])
                id_partner = choice(setxor1d(array(range(self.pop_size)), array([i])))
                arr_random = rand(self.problem_size)
                if pop[i][self.ID_FIT] < pop[id_partner][self.ID_FIT]:
                    temp += rand(self.problem_size) * (pop[i][self.ID_POS] - pop[id_partner][self.ID_POS])
                else:
                    temp += rand(self.problem_size) * (pop[id_partner][self.ID_POS] - pop[i][self.ID_POS])
                fit = self.get_fitness_position(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

                ### Batch size idea
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


class OriginalTLO(BaseTLO):
    """
    The original version of: Teaching Learning-based Optimization (TLO)
        Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems
    This is slower version which inspired from this version:
        https://github.com/andaviaco/tblo
    Notes:
        + I removed third loop to make it faster
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        BaseTLO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, kwargs=kwargs)

    def train(self):
        pop = [self.create_solution(minmax=0) for _ in range(self.pop_size)]
        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Teaching Phrase
                TF = randint(1, 3)  # 1 or 2 (never 3)
                best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

                #### Remove third loop here
                list_pos = array([item[self.ID_POS] for item in pop])
                pos_new = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * (best[self.ID_POS] - TF * mean(list_pos, axis=0))

                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

                ## Learning Phrase
                id_partner = choice(setxor1d(array(range(self.pop_size)), array([i])))

                #### Remove third loop here
                if pop[i][self.ID_FIT] < pop[id_partner][self.ID_FIT]:
                    diff = pop[i][self.ID_POS] - pop[id_partner][self.ID_POS]
                else:
                    diff = pop[id_partner][self.ID_POS] - pop[i][self.ID_POS]
                pos_new = pop[i][self.ID_POS] + uniform(0, 1, self.problem_size) * diff

                fit = self.get_fitness_position(pos_new)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit]

            best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
            self.loss_train.append(best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, best[self.ID_FIT]))
        self.solution = best
        return best[self.ID_POS], best[self.ID_FIT], self.loss_train