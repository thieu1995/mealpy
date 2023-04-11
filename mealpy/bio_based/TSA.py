#!/usr/bin/env python
# Created by "Thieu" at 17:23, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTSA(Optimizer):
    """
    The original version: Tunicate Swarm Algorithm (TSA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0952197620300385?via%3Dihub
        2. https://www.mathworks.com/matlabcentral/fileexchange/75182-tunicate-swarm-algorithm-tsa

    Notes:
        1. This algorithm has some limitations
        2. The paper has several wrong equations in algorithm
        3. The implementation in Matlab code has some difference to the paper
        4. This algorithm shares some similarities with the Barnacles Mating Optimizer (BMO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.bio_based.TSA import OriginalTSA
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
    >>> model = OriginalTSA(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired
    based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.
    """

    ID_WEI = 2

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pmin, pmax = 1, 4
        pop_new = []
        for idx in range(0, self.pop_size):
            c3 = np.random.random(self.problem.n_dims)
            c2 = np.random.random(self.problem.n_dims)
            c1 = np.random.random(self.problem.n_dims)
            M = np.fix(pmin + np.random.rand() * (pmax - pmin))
            A = (c2 + c3 - 2 * c1) / M
            t1 = self.g_best[self.ID_POS] + A * np.abs(self.g_best[self.ID_POS] - c2 * self.pop[idx][self.ID_POS])
            t2 = self.g_best[self.ID_POS] - A * np.abs(self.g_best[self.ID_POS] - c2 * self.pop[idx][self.ID_POS])
            new_pos = np.where(c3 >= 0.5, t1, t2)
            if idx != 0:
                new_pos = (new_pos + self.pop[idx-1][self.ID_POS]) / 2
            new_pos = self.amend_position(new_pos, self.problem.lb, self.problem.ub)
            pop_new.append([new_pos, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(new_pos)
        self.pop = self.update_target_wrapper_population(pop_new)
