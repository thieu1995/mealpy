# !/usr/bin/env python
# Created by "Thieu" at 18:41, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseEHO(Optimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)

    Links:
        1. https://doi.org/10.1109/ISCBI.2015.8

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + alpha (float): [0.3, 0.8], a factor that determines the influence of the best in each clan, default=0.5
        + beta (float): [0.3, 0.8], a factor that determines the influence of the x_center, default=0.5
        + n_clans (int): [3, 10], the number of clans, default=5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.EHO import BaseEHO
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
    >>> alpha = 0.5
    >>> beta = 0.5
    >>> n_clans = 5
    >>> model = BaseEHO(problem_dict1, epoch, pop_size, alpha, beta, n_clans)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
    In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, alpha=0.5, beta=0.5, n_clans=5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan, default=0.5
            beta (float): a factor that determines the influence of the x_center, default=0.5
            n_clans (int): the number of clans, default=5
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 3.0))
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.n_clans = self.validator.check_int("n_clans", n_clans, [2, int(self.pop_size/5)])
        self.n_individuals = int(self.pop_size / self.n_clans)
        self.nfe_per_epoch = self.pop_size + self.n_clans
        self.sort_flag = False

    def _create_pop_group(self, pop):
        pop_group = []
        for i in range(0, self.n_clans):
            group = pop[i * self.n_individuals: (i + 1) * self.n_individuals]
            pop_group.append(deepcopy(group))
        return pop_group

    def initialization(self):
        self.pop = self.create_population(self.pop_size)
        self.pop_group = self._create_pop_group(self.pop)
        _, self.g_best = self.get_global_best_solution(self.pop)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Clan updating operator
        pop_new = []
        for i in range(0, self.pop_size):
            clan_idx = int(i / self.n_individuals)
            pos_clan_idx = int(i % self.n_individuals)

            if pos_clan_idx == 0:  # The best in clan, because all clans are sorted based on fitness
                center = np.mean(np.array([item[self.ID_POS] for item in self.pop_group[clan_idx]]), axis=0)
                pos_new = self.beta * center
            else:
                pos_new = self.pop_group[clan_idx][pos_clan_idx][self.ID_POS] + self.alpha * np.random.uniform() * \
                          (self.pop_group[clan_idx][0][self.ID_POS] - self.pop_group[clan_idx][pos_clan_idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
        # Update fitness value
        self.pop = self.update_target_wrapper_population(pop_new)
        self.pop_group = self._create_pop_group(self.pop)

        # Separating operator
        for i in range(0, self.n_clans):
            self.pop_group[i], _ = self.get_global_best_solution(self.pop_group[i])
            self.pop_group[i][-1] = self.create_solution(self.problem.lb, self.problem.ub)
        self.pop = [agent for pack in self.pop_group for agent in pack]
