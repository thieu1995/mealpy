#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:33, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import multiply
from numpy.random import uniform, choice
from copy import deepcopy
from mealpy.root import RootAlgo


class BaseGA(RootAlgo):
    """
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """

    def __init__(self, root_algo_paras=None, ga_paras=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = ga_paras["epoch"]
        self.pop_size = ga_paras["pop_size"]
        self.pc = ga_paras["pc"]
        self.pm = ga_paras["pm"]

    ### Selection
    def _get_index_roulette_wheel_selection__(self, list_fitness, sum_fitness):
        # Just for positive numbers
        r = uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r = r + f
            if r > sum_fitness:
                return idx

    def _get_parents_kway_tournament_selection__(self, pop=None, k_way=10):
        list_id = choice(range(self.pop_size), k_way, replace=False)
        list_parents = [pop[i] for i in list_id]
        list_parents = sorted(list_parents, key=lambda temp: temp[self.ID_FIT])
        return list_parents[0:2]

    ### Crossover
    def _crossover_arthmetic_recombination__(self, dad=None, mom=None):
        r = uniform()  # w1 = w2 when r =0.5
        w1 = multiply(r, dad) + multiply((1 - r), mom)
        w2 = multiply(r, mom) + multiply((1 - r), dad)
        return w1, w2

    ### Mutation
    def _mutation_flip_point__(self, parent, index):
        w = deepcopy(parent)
        w[index] = uniform(self.domain_range[0], self.domain_range[1])
        return w

    def _create_next_generation__(self, pop):
        next_population = []
        while (len(next_population) < self.pop_size):
            ### Selection
            c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=10)
            w1, w2 = deepcopy(c1[0]), deepcopy(c2[0])
            ### Crossover
            if uniform() < self.pc:
                w1, w2 = self._crossover_arthmetic_recombination__(c1[self.ID_POS], c2[self.ID_POS])

            ### Mutation
            for idx in range(0, self.problem_size):
                if uniform() < self.pm:
                    w1 = self._mutation_flip_point__(w1, idx)
                if uniform() < self.pm:
                    w2 = self._mutation_flip_point__(w2, idx)

            c1_new = [deepcopy(w1), self._fitness_model__(w1, minmax=1)]
            c2_new = [deepcopy(w2), self._fitness_model__(w2, minmax=1)]
            next_population.append(c1_new)
            next_population.append(c2_new)
        return next_population

    def _train__(self):
        pop = [self._create_solution__(minmax=0) for _ in range(self.pop_size)]
        g_best = self._get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Next generations
            pop = deepcopy(self._create_next_generation__(pop))

            # update global best solution
            g_best = self._update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], self.loss_train
