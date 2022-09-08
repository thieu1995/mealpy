# !/usr/bin/env python
# Created by "Thieu" at 12:09, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCA(Optimizer):
    """
    The original version of: Culture Algorithm (CA)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + accepted_rate (float): [0.1, 0.5], probability of accepted rate, default: 0.15

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.human_based.CA import OriginalCA
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
    >>> accepted_rate = 0.15
    >>> model = OriginalCA(epoch, pop_size, accepted_rate)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, B., Zhao, L. and Lu, J.H., 2009, April. Wind power forecast using RBF network and culture algorithm.
    In 2009 International Conference on Sustainable Power Generation and Supply (pp. 1-6). IEEE.
    """

    def __init__(self, epoch=10000, pop_size=100, accepted_rate=0.15, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            accepted_rate (float): probability of accepted rate, default: 0.15
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.accepted_rate = self.validator.check_float("accepted_rate", accepted_rate, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "accepted_rate"])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def initialize_variables(self):
        ## Dynamic variables
        self.dyn_belief_space = {
            "lb": self.problem.lb,
            "ub": self.problem.ub,
        }
        self.dyn_accepted_num = int(self.accepted_rate * self.pop_size)
        # update situational knowledge (g_best here is a element inside belief space)

    def create_faithful__(self, lb, ub):
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def update_belief_space__(self, belief_space, pop_accepted):
        pos_list = np.array([solution[self.ID_POS] for solution in pop_accepted])
        belief_space["lb"] = np.min(pos_list, axis=0)
        belief_space["ub"] = np.max(pos_list, axis=0)
        return belief_space

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # create next generation
        pop_child = [self.create_faithful__(self.dyn_belief_space["lb"], self.dyn_belief_space["ub"]) for _ in range(0, self.pop_size)]

        # select next generation
        pop_new = []
        pop_full = self.pop + pop_child
        size_new = len(pop_full)
        for _ in range(0, self.pop_size):
            id1, id2 = np.random.choice(list(range(0, size_new)), 2, replace=False)
            pop_new.append(self.get_better_solution(pop_full[id1], pop_full[id2]))
        self.pop = self.get_sorted_strim_population(pop_new)

        # Get accepted faithful
        accepted = self.pop[:self.dyn_accepted_num]

        # Update belief_space
        self.dyn_belief_space = self.update_belief_space__(self.dyn_belief_space, accepted)
