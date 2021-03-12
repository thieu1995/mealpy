#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:22, 11/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array
from numpy.random import uniform, choice
from copy import deepcopy
from mealpy.root import Root


class BaseMA(Root):
    """
        The original verion of: Memetic Algorithm (MA)
            (On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms)
        Link:
            Clever Algorithms: Nature-Inspired Programming Recipes - Memetic Algorithm (MA)
            http://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_BIT = 2

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 pc=0.98, pm=0.025, p_local=0.5, max_local_gens=10, bits_per_param=16, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.p_local = p_local
        self.max_local_gens = max_local_gens
        self.bits_per_param = bits_per_param
        self.bits_total = self.problem_size * self.bits_per_param

    def create_solution(self, minmax=0):
        solution = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=solution, minmax=minmax)
        bitstring = self._random_bitstring__()
        return [solution, fitness, bitstring]

    def _random_bitstring__(self):
        bitstring = ["1" if uniform() < 0.5 else "0" for _ in range(0, self.bits_total)]
        my_string = ''
        return my_string.join(bitstring)

    def _decode__(self, bitstring=None):
        # "11000000100101000101010" - 32 bit for 2 variable. eg. x1 and x2
        # bits_per_param = 16
        ## Decode the random bitstring into real number
        vector = []
        for idx in range(0, self.problem_size):
            param = bitstring[idx * self.bits_per_param: (idx + 1) * self.bits_per_param]  # Select 16 bit every time
            temp = self.lb[idx] + ((self.ub[idx] - self.lb[idx]) / ((2.0 ** self.bits_per_param) - 1)) * int(param, 2)
            vector.append(temp)
        return array(vector)

    def _binary_tournament__(self, pop=None):
        i1, i2 = choice(range(0, self.pop_size), 2, replace=False)
        pos_winner = pop[i1] if pop[i1][self.ID_FIT] < pop[i2][self.ID_FIT] else pop[i2]
        return deepcopy(pos_winner)

    def _point_mutation__(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def _crossover__(self, dad=None, mom=None):
        if uniform() >= self.pc:
            return deepcopy(dad)
        else:
            child = ""
            for idx in range(0, self.bits_total):
                if uniform() < 0.5:
                    child += dad[idx]
                else:
                    child += mom[idx]
            return child

    def _reproduce__(self, pop=None):
        children = deepcopy(pop)
        for idx in range(0, self.pop_size):
            ancient = pop[idx + 1] if idx % 2 == 0 else pop[idx - 1]
            if idx == self.pop_size - 1:
                ancient = pop[0]
            bitstring_new = self._crossover__(pop[idx][self.ID_BIT], ancient[self.ID_BIT])
            bitstring_new = self._point_mutation__(bitstring_new)
            pos_new = self._decode__(bitstring_new)
            fit = self.get_fitness_position(pos_new)
            children[idx] = [pos_new, fit, bitstring_new]
        return children

    def _bits_climber__(self, child=None):
        current = deepcopy(child)
        for idx in range(0, self.max_local_gens):
            bitstring_new = self._point_mutation__(child[self.ID_BIT])
            pos_new = self._decode__(bitstring_new)
            fit = self.get_fitness_position(pos_new)
            if fit < child[self.ID_FIT]:
                current = [pos_new, fit, bitstring_new]
        return current

    def train(self):
        pop = [self.create_solution(minmax=self.ID_MIN_PROB) for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            # Next generations
            pop = [self._binary_tournament__(pop) for _ in range(self.pop_size)]
            pop = self._reproduce__(pop)

            # Searching in local
            for i in range(0, self.pop_size):
                if uniform() < self.p_local:
                    pop[i] = self._bits_climber__(pop[i])

            # update global best position
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
