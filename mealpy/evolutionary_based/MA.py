#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:22, 11/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import time
import numpy as np
from mealpy.optimizer import Optimizer


class BaseMA(Optimizer):
    """
        The original version of: Memetic Algorithm (MA)
            (On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms)
        Link:
            Clever Algorithms: Nature-Inspired Programming Recipes - Memetic Algorithm (MA)
            http://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_BIT = 2

    def __init__(self, problem: dict, epoch=1000, pop_size=100, pc=0.98, pm=0.025, p_local=0.5, max_local_gens=10, bits_per_param=16):
        """
        Args:
            problem (dict): a dictionary of your problem
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
            p_local ():
            max_local_gens ():
            bits_per_param ():
        """
        super().__init__(problem)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.p_local = p_local
        self.max_local_gens = max_local_gens
        self.bits_per_param = bits_per_param
        self.bits_total = self.problem_size * self.bits_per_param

    def create_solution(self):
        position = np.random.uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position)
        bitstring = ''.join(["1" if np.random.uniform() < 0.5 else "0" for _ in range(0, self.bits_total)])
        return [position, fitness, bitstring]

    def _decode__(self, bitstring=None):
        """
        Decode the random bitstring into real number
        Args:
            bitstring (str): "11000000100101000101010" - bits_per_param = 16, 32 bit for 2 variable. eg. x1 and x2

        Returns:
            list of real number (vector)
        """
        vector = np.ones(self.problem_size)
        for idx in range(0, self.problem_size):
            param = bitstring[idx * self.bits_per_param: (idx + 1) * self.bits_per_param]  # Select 16 bit every time
            vector[idx] = self.lb[idx] + ((self.ub[idx] - self.lb[idx]) / ((2.0 ** self.bits_per_param) - 1)) * int(param, 2)
        return vector

    def _crossover__(self, dad=None, mom=None):
        if np.random.uniform() >= self.pc:
            temp = [dad].copy()
            return temp[0]
        else:
            child = ""
            for idx in range(0, self.bits_total):
                if np.random.uniform() < 0.5:
                    child += dad[idx]
                else:
                    child += mom[idx]
            return child

    def _point_mutation__(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if np.random.uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def create_next_generation(self, pop: list):
        ## Binary tournament
        children = [self.get_solution_kway_tournament_selection(pop, k_way=2, output=1)[0] for _ in range(self.pop_size)]

        ## Reproduction
        for idx in range(0, self.pop_size):
            ancient = pop[idx + 1] if idx % 2 == 0 else pop[idx - 1]
            if idx == self.pop_size - 1:
                ancient = pop[0]
            bitstring_new = self._crossover__(pop[idx][self.ID_BIT], ancient[self.ID_BIT])
            bitstring_new = self._point_mutation__(bitstring_new)
            pos_new = self._decode__(bitstring_new)
            fit_new = self.get_fitness_position(pos_new)
            children[idx] = [pos_new, fit_new, bitstring_new]
        return children

    def _bits_climber__(self, child=None):
        current = child.copy()
        for idx in range(0, self.max_local_gens):
            child = current.copy()
            bitstring_new = self._point_mutation__(child[self.ID_BIT])
            pos_new = self._decode__(bitstring_new)
            fit_new = self.get_fitness_position(pos_new)
            current = self.get_better_solution(child, [pos_new, fit_new, bitstring_new])
        return current

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_global_best_solution(pop)
        self.history_list_g_best = [g_best]
        self.history_list_c_best = self.history_list_g_best.copy()

        for epoch in range(0, self.epoch):
            time_start = time.time()

            # Create next generations
            pop = self.create_next_generation(pop)

            # Searching in local
            for i in range(0, self.pop_size):
                if np.random.uniform() < self.p_local:
                    pop[i] = self._bits_climber__(pop[i])

            # Sort the population and update the global best solution
            pop = self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_start = time.time() - time_start
            self.history_list_epoch_time.append(time_start)
            self.print_epoch(epoch + 1, time_start)
            self.history_list_pop.append(pop.copy())

        ## Additional information for the framework
        self.solution = self.history_list_g_best[-1]
        self.save_data()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]

