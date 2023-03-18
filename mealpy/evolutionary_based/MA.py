#!/usr/bin/env python
# Created by "Thieu" at 14:22, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalMA(Optimizer):
    """
    The original version of: Memetic Algorithm (MA)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.85
        + pm (float): [0.05, 0.3], mutation probability, default = 0.15
        + p_local (float): [0.3, 0.7], Probability of local search for each agent, default=0.5
        + max_local_gens (int): [5, 25], number of local search agent will be created during local search mechanism, default=10
        + bits_per_param (int): [2, 4, 8, 16], number of bits to decode a real number to 0-1 bitstring, default=4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.MA import OriginalMA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> pc = 0.85
    >>> pm = 0.15
    >>> p_local = 0.5
    >>> max_local_gens = 10
    >>> bits_per_param = 4
    >>> model = OriginalMA(epoch, pop_size, pc, pm, p_local, max_local_gens, bits_per_param)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Moscato, P., 1989. On evolution, search, optimization, genetic algorithms and martial arts:
    Towards memetic algorithms. Caltech concurrent computation program, C3P Report, 826, p.1989.
    """

    ID_BIT = 2

    def __init__(self, epoch=10000, pop_size=100, pc=0.85, pm=0.15,
                 p_local=0.5, max_local_gens=10, bits_per_param=4, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.85
            pm (float): mutation probability, default = 0.15
            p_local (float): Probability of local search for each agent, default=0.5
            max_local_gens (int): Number of local search agent will be created during local search mechanism, default=10
            bits_per_param (int): Number of bits to decode a real number to 0-1 bitstring, default=4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.p_local = self.validator.check_float("p_local", p_local, (0, 1.0))
        self.max_local_gens = self.validator.check_int("max_local_gens", max_local_gens, [2, int(pop_size/2)])
        self.bits_per_param = self.validator.check_int("bits_per_param", bits_per_param, [2, 32])
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "p_local", "max_local_gens", "bits_per_param"])
        self.sort_flag = True

    def initialize_variables(self):
        self.bits_total = self.problem.n_dims * self.bits_per_param

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, bitstring]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        bitstring = ''.join(["1" if np.random.uniform() < 0.5 else "0" for _ in range(0, self.bits_total)])
        return [position, target, bitstring]

    def decode_(self, bitstring=None):
        """
        Decode the random bitstring into real number

        Args:
            bitstring (str): "11000000100101000101010" - bits_per_param = 16, 32 bit for 2 variable. eg. x1 and x2

        Returns:
            list: list of real number (vector)
        """
        vector = np.ones(self.problem.n_dims)
        for idx in range(0, self.problem.n_dims):
            param = bitstring[idx * self.bits_per_param: (idx + 1) * self.bits_per_param]  # Select 16 bit every time
            vector[idx] = self.problem.lb[idx] + ((self.problem.ub[idx] - self.problem.lb[idx]) / ((2.0 ** self.bits_per_param) - 1)) * int(param, 2)
        return vector

    def crossover__(self, dad=None, mom=None):
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

    def point_mutation__(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if np.random.uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def bits_climber__(self, child=None):
        current = deepcopy(child)
        list_local = []
        for idx in range(0, self.max_local_gens):
            child = deepcopy(current)
            bitstring_new = self.point_mutation__(child[self.ID_BIT])
            pos_new = self.decode_(bitstring_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            list_local.append([pos_new, None, bitstring_new])
            if self.mode not in self.AVAILABLE_MODES:
                list_local[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        list_local = self.update_target_wrapper_population(list_local)
        list_local.append(child)
        _, best = self.get_global_best_solution(list_local)
        return best

    def create_child__(self, idx, pop_copy):
        ancient = pop_copy[idx + 1] if idx % 2 == 0 else pop_copy[idx - 1]
        if idx == self.pop_size - 1:
            ancient = pop_copy[0]
        bitstring_new = self.crossover__(pop_copy[idx][self.ID_BIT], ancient[self.ID_BIT])
        bitstring_new = self.point_mutation__(bitstring_new)
        pos_new = self.decode_(bitstring_new)
        pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
        target = self.get_target_wrapper(pos_new)
        return [pos_new, target, bitstring_new]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Binary tournament
        children = []
        for idx in range(0, self.pop_size):
            idx_offspring = self.get_index_kway_tournament_selection(self.pop, k_way=2, output=1)[0]
            children.append(deepcopy(self.pop[idx_offspring]))
        pop = []
        for idx in range(0, self.pop_size):
            ancient = children[idx + 1] if idx % 2 == 0 else children[idx - 1]
            if idx == self.pop_size - 1:
                ancient = children[0]
            bitstring_new = self.crossover__(children[idx][self.ID_BIT], ancient[self.ID_BIT])
            bitstring_new = self.point_mutation__(bitstring_new)
            pos_new = self.decode_(bitstring_new)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None, bitstring_new])
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop = self.update_target_wrapper_population(pop)

        # Searching in local
        for i in range(0, self.pop_size):
            if np.random.rand() < self.p_local:
                self.pop[i] = self.bits_climber__(pop[i])
