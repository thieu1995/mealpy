#!/usr/bin/env python
# Created by "Thieu" at 14:07, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseICA(Optimizer):
    """
    The original version of: Imperialist Competitive Algorithm (ICA)

    Links:
        1. https://ieeexplore.ieee.org/document/4425083

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + empire_count (int): [3, 10], Number of Empires (also Imperialists)
        + assimilation_coeff (float): [1.0, 3.0], Assimilation Coefficient (beta in the paper)
        + revolution_prob (float): [0.01, 0.1], Revolution Probability
        + revolution_rate (float): [0.05, 0.2], Revolution Rate       (mu)
        + revolution_step_size (float): [0.05, 0.2], Revolution Step Size  (sigma)
        + zeta (float): [0.05, 0.2], Colonies Coefficient in Total Objective Value of Empires

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.ICA import BaseICA
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
    >>> empire_count = 5
    >>> assimilation_coeff = 1.5
    >>> revolution_prob = 0.05
    >>> revolution_rate = 0.1
    >>> revolution_step_size = 0.1
    >>> zeta = 0.1
    >>> model = BaseICA(problem_dict1, epoch, pop_size, empire_count, assimilation_coeff, revolution_prob, revolution_rate, revolution_step_size, zeta)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Atashpaz-Gargari, E. and Lucas, C., 2007, September. Imperialist competitive algorithm: an algorithm for
    optimization inspired by imperialistic competition. In 2007 IEEE congress on evolutionary computation (pp. 4661-4667). Ieee.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, empire_count=5, assimilation_coeff=1.5,
                 revolution_prob=0.05, revolution_rate=0.1, revolution_step_size=0.1, zeta=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (n: pop_size, m: clusters), default = 100
            empire_count (int): Number of Empires (also Imperialists)
            assimilation_coeff (float): Assimilation Coefficient (beta in the paper)
            revolution_prob (float): Revolution Probability
            revolution_rate (float): Revolution Rate       (mu)
            revolution_step_size (float): Revolution Step Size  (sigma)
            zeta (float): Colonies Coefficient in Total Objective Value of Empires
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.empire_count = self.validator.check_int("empire_count", empire_count, [2, 2 + int(self.pop_size / 5)])
        self.assimilation_coeff = self.validator.check_float("assimilation_coeff", assimilation_coeff, [1.0, 3.0])
        self.revolution_prob = self.validator.check_float("revolution_prob", revolution_prob, (0, 1.0))
        self.revolution_rate = self.validator.check_float("revolution_rate", revolution_rate, (0, 1.0))
        self.revolution_step_size = self.validator.check_float("revolution_step_size", revolution_step_size, (0, 1.0))
        self.zeta = self.validator.check_float("zeta", zeta, (0, 1.0))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True
        self.pop_empires, self.pop_colonies, self.empires = None, None, None
        self.n_revoluted_variables, self.idx_list_variables = None, None

    def revolution_country(self, position, idx_list_variables, n_revoluted):
        pos_new = position + self.revolution_step_size * np.random.normal(0, 1, self.problem.n_dims)
        idx_list = np.random.choice(idx_list_variables, n_revoluted, replace=False)
        position[idx_list] = pos_new[idx_list]  # Change only those selected index
        return position

    def initialization(self):
        pop = self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(pop)

        # Initialization
        self.n_revoluted_variables = int(round(self.revolution_rate * self.problem.n_dims))
        self.idx_list_variables = list(range(0, self.problem.n_dims))

        # pop = Empires
        colony_count = self.pop_size - self.empire_count
        self.pop_empires = deepcopy(self.pop[:self.empire_count])
        self.pop_colonies = deepcopy(self.pop[self.empire_count:])

        cost_empires_list = np.array([solution[self.ID_TAR][self.ID_FIT] for solution in self.pop_empires])
        cost_empires_list_normalized = cost_empires_list - (np.max(cost_empires_list) + np.min(cost_empires_list))
        prob_empires_list = np.abs(cost_empires_list_normalized / np.sum(cost_empires_list_normalized))
        # Randomly choose colonies to empires
        self.empires = {}
        idx_already_selected = []
        for i in range(0, self.empire_count - 1):
            self.empires[i] = []
            n_colonies = int(round(prob_empires_list[i] * colony_count))
            idx_list = np.random.choice(list(set(range(0, colony_count)) - set(idx_already_selected)), n_colonies, replace=False).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                self.empires[i].append(self.pop_colonies[idx])
        idx_last = list(set(range(0, colony_count)) - set(idx_already_selected))
        self.empires[self.empire_count - 1] = []
        for idx in idx_last:
            self.empires[self.empire_count - 1].append(self.pop_colonies[idx])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Assimilation
        for idx, colonies in self.empires.items():
            for idx_colony, colony in enumerate(colonies):
                pos_new = colony[self.ID_POS] + self.assimilation_coeff * \
                          np.random.uniform(0, 1, self.problem.n_dims) * (self.pop_empires[idx][self.ID_POS] - colony[self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                self.empires[idx][idx_colony][self.ID_POS] = pos_new
            self.empires[idx] = self.update_target_wrapper_population(self.empires[idx])

        # Revolution
        for idx, colonies in self.empires.items():
            # Apply revolution to Imperialist
            pos_new = self.revolution_country(self.pop_empires[idx][self.ID_POS], self.idx_list_variables, self.n_revoluted_variables)
            self.pop_empires[idx][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            # Apply revolution to Colonies
            for idx_colony, colony in enumerate(colonies):
                if np.random.rand() < self.revolution_prob:
                    pos_new = self.revolution_country(colony[self.ID_POS], self.idx_list_variables, self.n_revoluted_variables)
                    self.empires[idx][idx_colony][self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            self.empires[idx] = self.update_target_wrapper_population(self.empires[idx])
        self.pop_empires = self.update_target_wrapper_population(self.pop_empires)
        _, g_best = self.update_global_best_solution(self.pop_empires, save=False)

        # Intra-Empire Competition
        for idx, colonies in self.empires.items():
            for idx_colony, colony in enumerate(colonies):
                if self.compare_agent(colony, self.pop_empires[idx]):
                    self.empires[idx][idx_colony], self.pop_empires[idx] = deepcopy(self.pop_empires[idx]), deepcopy(colony)

        # Update Total Objective Values of Empires
        cost_empires_list = []
        for idx, colonies in self.empires.items():
            fit_list = np.array([solution[self.ID_TAR][self.ID_FIT] for solution in colonies])
            fit_empire = self.pop_empires[idx][self.ID_TAR][self.ID_FIT] + self.zeta * np.mean(fit_list)
            cost_empires_list.append(fit_empire)
        cost_empires_list = np.array(cost_empires_list)

        # Find possession probability of each empire based on its total power
        cost_empires_list_normalized = cost_empires_list - (np.max(cost_empires_list) + np.min(cost_empires_list))
        prob_empires_list = np.abs(cost_empires_list_normalized / np.sum(cost_empires_list_normalized))  # Vector P

        uniform_list = np.random.uniform(0, 1, len(prob_empires_list))  # Vector R
        vector_D = prob_empires_list - uniform_list
        idx_empire = np.argmax(vector_D)

        # Find the weakest empire and weakest colony inside it
        idx_weakest_empire = np.argmax(cost_empires_list)
        if len(self.empires[idx_weakest_empire]) > 0:
            colonies_sorted, best, worst = self.get_special_solutions(self.empires[idx_weakest_empire])
            self.empires[idx_empire].append(colonies_sorted.pop(-1))
        else:
            self.empires[idx_empire].append(self.pop_empires.pop(idx_weakest_empire))

        self.pop = self.pop_empires + self.pop_colonies

