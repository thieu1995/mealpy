#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:22, 11/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseMA(Optimizer):
    """
        The original version of: Memetic Algorithm (MA)
            (On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms)
        Link:
            Clever Algorithms: Nature-Inspired Programming Recipes - Memetic Algorithm (MA)
            http://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
    """
    ID_BIT = 2

    def __init__(self, problem, epoch=10000, pop_size=100, pc=0.85, pm=0.15,
                 p_local=0.5, max_local_gens=20, bits_per_param=16, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.85
            pm (float): mutation probability, default = 0.15
            p_local (float): Probability of local search for each agent, default=0.5
            max_local_gens (int): Number of local search agent will be created during local search mechanism, default=20
            bits_per_param (int): Number of bits to decode a real number to 0-1 bitstring, default=16
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.p_local = p_local
        self.max_local_gens = max_local_gens
        self.bits_per_param = bits_per_param
        self.bits_total = self.problem.n_dims * self.bits_per_param

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]], bitstring]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        bitstring = ''.join(["1" if np.random.uniform() < 0.5 else "0" for _ in range(0, self.bits_total)])
        return [position, fitness, bitstring]

    def _decode(self, bitstring=None):
        """
        Decode the random bitstring into real number
        Args:
            bitstring (str): "11000000100101000101010" - bits_per_param = 16, 32 bit for 2 variable. eg. x1 and x2

        Returns:
            list of real number (vector)
        """
        vector = np.ones(self.problem.n_dims)
        for idx in range(0, self.problem.n_dims):
            param = bitstring[idx * self.bits_per_param: (idx + 1) * self.bits_per_param]  # Select 16 bit every time
            vector[idx] = self.problem.lb[idx] + ((self.problem.ub[idx] - self.problem.lb[idx]) / ((2.0 ** self.bits_per_param) - 1)) * int(param, 2)
        return vector

    def _crossover(self, dad=None, mom=None):
        if np.random.uniform() >= self.pc:
            temp = deepcopy([dad])
            return temp[0]
        else:
            child = ""
            for idx in range(0, self.bits_total):
                if np.random.uniform() < 0.5:
                    child += dad[idx]
                else:
                    child += mom[idx]
            return child

    def _point_mutation(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if np.random.uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def _bits_climber(self, child=None):
        current = deepcopy(child)
        for idx in range(0, self.max_local_gens):
            child = deepcopy(current)
            bitstring_new = self._point_mutation(child[self.ID_BIT])
            pos_new = self._decode(bitstring_new)
            fit_new = self.get_fitness_position(pos_new)
            current = self.get_better_solution(child, [pos_new, fit_new, bitstring_new])
        return current

    def create_child(self, idx, pop_copy):
        ancient = pop_copy[idx + 1] if idx % 2 == 0 else pop_copy[idx - 1]
        if idx == self.pop_size - 1:
            ancient = pop_copy[0]
        bitstring_new = self._crossover(pop_copy[idx][self.ID_BIT], ancient[self.ID_BIT])
        bitstring_new = self._point_mutation(bitstring_new)
        pos_new = self._decode(bitstring_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new, bitstring_new]

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = self.pop_size
        ## Binary tournament
        children = [self.get_solution_kway_tournament_selection(self.pop, k_way=2, output=1)[0] for _ in range(self.pop_size)]
        pop = []
        for idx in range(0, self.pop_size):
            ancient = children[idx + 1] if idx % 2 == 0 else children[idx - 1]
            if idx == self.pop_size - 1:
                ancient = children[0]
            bitstring_new = self._crossover(children[idx][self.ID_BIT], ancient[self.ID_BIT])
            bitstring_new = self._point_mutation(bitstring_new)
            pos_new = self._decode(bitstring_new)
            pop.append([pos_new, None, bitstring_new])
        self.pop = self.update_fitness_population(pop)

        # Searching in local
        for i in range(0, self.pop_size):
            if np.random.rand() < self.p_local:
                self.pop[i] = self._bits_climber(pop[i])
                nfe_epoch += self.max_local_gens
        self.nfe_per_epoch = nfe_epoch
