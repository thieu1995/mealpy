#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalDMOA(Optimizer):
    """
    The original version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2022.114570
        2. https://www.mathworks.com/matlabcentral/fileexchange/105125-dwarf-mongoose-optimization-algorithm

    Notes:
        1. Matlab code is litle bit difference than original paper
        2. There are some meaningless parameters and equations in the matlab code
        3. Weak algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.DMOA import OriginalDMOA
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
    >>> n_baby_sitter = 3
    >>> peep = 2
    >>> model = OriginalDMOA(epoch, pop_size, n_baby_sitter, peep)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Agushaka, J. O., Ezugwu, A. E., & Abualigah, L. (2022). Dwarf mongoose optimization algorithm.
    Computer methods in applied mechanics and engineering, 391, 114570.
    """

    def __init__(self, epoch=10000, pop_size=100, n_baby_sitter=3, peep=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_baby_sitter = self.validator.check_int("n_baby_sitter", n_baby_sitter, [2, 10])
        self.peep = self.validator.check_float("peep", peep, [1, 10.])
        self.n_scout = self.pop_size - self.n_baby_sitter
        self.support_parallel_modes = False
        self.set_parameters(["epoch", "pop_size", "n_baby_sitter", "peep"])
        self.sort_flag = False

    def initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.tau = -np.inf
        self.L = np.round(0.6 * self.problem.n_dims * self.n_baby_sitter)


    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Abandonment Counter
        CF = (1 - (epoch+1)/self.epoch)**(2*(epoch+1)/self.epoch)
        fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)
        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = np.random.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha][self.ID_POS] + phi * (self.pop[alpha][self.ID_POS] - self.pop[k][self.ID_POS])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                self.pop[idx] = [new_pos, new_tar]
            else:
                self.C[idx] += 1

        SM = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx][self.ID_POS] + phi * (self.pop[idx][self.ID_POS] - self.pop[k][self.ID_POS])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            ## Sleeping mould
            SM[idx] = (new_tar[self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT])/np.max([new_tar[self.ID_FIT], self.pop[idx][self.ID_TAR][self.ID_FIT]])
            if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                self.pop[idx] = [new_pos, new_tar]
            else:
                self.C[idx] += 1
        ## Baby sitters
        for idx in range(0, self.n_baby_sitter):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
                self.C[idx] = 0

        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, self.pop_size):
            M = SM[idx] * self.pop[idx][self.ID_POS] / self.pop[idx][self.ID_POS]
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            if new_tau > self.tau:
                new_pos = self.pop[idx][self.ID_POS] - CF * phi * np.random.rand() * (self.pop[idx][self.ID_POS] - M)
            else:
                new_pos = self.pop[idx][self.ID_POS] + CF * phi * np.random.rand() * (self.pop[idx][self.ID_POS] - M)
            self.tau = new_tau
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            self.pop[idx] = [new_pos, new_tar]


class DevDMOA(Optimizer):
    """
    The developed version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Notes:
        1. Removed the parameter n_baby_sitter
        2. Changed in section # Next Mongoose position
        3. Removed the meaningless variable tau

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.DMOA import DevDMOA
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
    >>> peep = 2
    >>> model = DevDMOA(epoch, pop_size, peep)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, peep=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.peep = self.validator.check_float("peep", peep, [1, 10.])
        self.set_parameters(["epoch", "pop_size", "peep"])
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.L = np.round(0.6 * self.epoch)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Abandonment Counter
        CF = (1 - (epoch + 1) / self.epoch) ** (2 * (epoch + 1) / self.epoch)
        fit_list = np.array([agent[self.ID_TAR][self.ID_FIT] for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)

        ## Foraging led by Alpha female
        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = np.random.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha][self.ID_POS] + phi * (self.pop[alpha][self.ID_POS] - self.pop[k][self.ID_POS])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                self.pop[idx] = [new_pos, new_tar]
            else:
                self.C[idx] += 1

        ## Scout group
        SM = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = np.random.choice(list(set(range(0, self.pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx][self.ID_POS] + phi * (self.pop[idx][self.ID_POS] - self.pop[k][self.ID_POS])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            ## Sleeping mould
            SM[idx] = (new_tar[self.ID_FIT] - self.pop[idx][self.ID_TAR][self.ID_FIT]) / np.max([new_tar[self.ID_FIT], self.pop[idx][self.ID_TAR][self.ID_FIT]])
            if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                self.pop[idx] = [new_pos, new_tar]
            else:
                self.C[idx] += 1

        ## Baby sitters
        for idx in range(0, self.pop_size):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.create_solution(self.problem.lb, self.problem.ub)
                self.C[idx] = 0

        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, self.pop_size):
            phi = (self.peep / 2) * np.random.uniform(-1, 1, self.problem.n_dims)
            if new_tau > SM[idx]:
                new_pos = self.g_best[self.ID_POS] - CF * phi * (self.g_best[self.ID_POS] - SM[idx] * self.pop[idx][self.ID_POS])
            else:
                new_pos = self.pop[idx][self.ID_POS] + CF * phi * (self.g_best[self.ID_POS] - SM[idx] * self.pop[idx][self.ID_POS])
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            new_tar = self.get_target_wrapper(new_pos)
            if self.compare_agent([new_pos, new_tar], self.pop[idx]):
                self.pop[idx] = [new_pos, new_tar]
