#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:14, 18/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint, choice, rand
from numpy import array, mean, setxor1d
from copy import deepcopy
from mealpy.root import Root

class BaseTLO(Root):
    """
    An elitist teaching-learning-based optimization algorithm for solving complex constrained optimization problems(TLO)
        This is my version taken the advantages of numpy array to faster handler operations.
    """
    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_paras)
        self.epoch =  epoch
        self.pop_size = pop_size

    def _calculate_mean__(self, pop=None):
        temp = mean(array([item[self.ID_POS] for item in pop]), axis=0)
        return temp

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Teaching Phrase
                TF = randint(1, 3)  # 1 or 2 (never 3)
                MEAN = self._calculate_mean__(pop)
                arr_random = rand(self.problem_size)
                DIFF_MEAN = arr_random * (g_best[self.ID_POS] - TF * MEAN)
                temp = pop[i][self.ID_POS] + DIFF_MEAN
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

                ## Learning Phrase
                temp = deepcopy(pop[i][self.ID_POS])
                id_partner = choice(setxor1d(array(range(self.pop_size)), array([i])))
                arr_random = rand(self.problem_size)
                if pop[i][self.ID_FIT] < pop[id_partner][self.ID_FIT]:
                    temp += arr_random * (pop[i][self.ID_POS] - pop[id_partner][self.ID_POS])
                else:
                    temp += arr_random * (pop[id_partner][self.ID_POS] - pop[i][self.ID_POS])
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_FIT], g_best[self.ID_FIT], self.loss_train


class OriginalTLO(BaseTLO):
    """
    Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems (TLO)
    This is slower version which inspired from this version:
        https://github.com/andaviaco/tblo
    """

    def __init__(self, root_paras=None, epoch=750, pop_size=100):
        BaseTLO.__init__(self, root_paras, epoch, pop_size)

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        for epoch in range(self.epoch):
            for i in range(self.pop_size):

                ## Teaching Phrase
                TF = randint(1, 3)  # 1 or 2 (never 3)
                best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
                temp = deepcopy(pop[i][self.ID_POS])
                for j in range(self.problem_size):
                    s_mean = mean([item[self.ID_POS][j] for item in pop])
                    r = uniform()
                    diff_mean = best[self.ID_POS][j] - TF * s_mean
                    temp[j] = pop[i][self.ID_POS][j] + r * diff_mean
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

                ## Learning Phrase
                temp = deepcopy(pop[i][self.ID_POS])
                id_partner = choice(setxor1d(array(range(self.pop_size)), array([i])))
                for j in range(self.problem_size):
                    if pop[i][self.ID_FIT] < pop[id_partner][self.ID_FIT]:
                        diff = pop[i][self.ID_POS][j] - pop[id_partner][self.ID_POS][j]
                    else:
                        diff = pop[id_partner][self.ID_POS][j] - pop[i][self.ID_POS][j]
                    r = uniform()
                    temp[j] = pop[i][self.ID_POS][j] + r * diff
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            best = self._get_global_best__(pop=pop, id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROB)
            self.loss_train.append(best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, best[self.ID_FIT]))

        return best[self.ID_FIT], best[self.ID_FIT], self.loss_train