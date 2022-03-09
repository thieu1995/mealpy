# !/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseMFO(Optimizer):
    """
    My changed version of: Moth-Flame Optimization (MFO)

    Notes
    ~~~~~
    + The flow of algorithm is changed
    + The old solution is updated

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MFO import BaseMFO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = BaseMFO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-based systems, 89, pp.228-249.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (epoch + 1) * ((-1) / self.epoch)

        pop_flames, g_best = self.get_global_best_solution(self.pop)

        pop_new = []
        for idx in range(0, self.pop_size):
            #   D in Eq.(3.13)
            distance_to_flame = np.abs(pop_flames[idx][self.ID_POS] - self.pop[idx][self.ID_POS])
            t = (a - 1) * np.random.uniform(0, 1, self.problem.n_dims) + 1
            b = 1

            # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
            temp_1 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[idx][self.ID_POS]

            # Update the position of the moth with respect to one flame Eq.(3.12).
            ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
            temp_2 = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + g_best[self.ID_POS]

            list_idx = idx * np.ones(self.problem.n_dims)
            pos_new = np.where(list_idx < num_flame, temp_1, temp_2)

            ## This is the way I make this algorithm working. I tried to run matlab code with large dimension and it doesn't convergence.
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalMFO(BaseMFO):
    """
    The original version of: Moth-flame Optimization (MFO)

    Link:
        1. https://www.mathworks.com/matlabcentral/fileexchange/52269-moth-flame-optimization-mfo-algorithm

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.MFO import OriginalMFO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalMFO(problem_dict1, epoch, pop_size)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. Moth-flame optimization algorithm: A novel nature-inspired
    heuristic paradigm. Knowledge-based systems, 89, pp.228-249.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(problem, epoch, pop_size, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (epoch + 1) * ((-1) / self.epoch)

        pop_flames, g_best = self.get_global_best_solution(self.pop)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(self.problem.n_dims):
                #   D in Eq.(3.13)
                distance_to_flame = np.abs(pop_flames[idx][self.ID_POS][j] - self.pop[idx][self.ID_POS][j])
                t = (a - 1) * np.random.uniform() + 1
                b = 1
                if idx <= num_flame:  # Update the position of the moth with respect to its corresponding flame
                    # Eq.(3.12)
                    pos_new[j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[idx][self.ID_POS][j]
                else:  # Update the position of the moth with respect to one flame
                    # Eq.(3.12).
                    ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
                    pos_new[j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + pop_flames[num_flame][self.ID_POS][j]
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        self.pop = self.update_fitness_population(pop_new)
