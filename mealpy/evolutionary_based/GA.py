# !/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.GA import BaseGA
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "obj_func": fitness_function,
    >>>     "n_dims": 5,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> pc = 0.9
    >>> pm = 0.05
    >>> model = BaseGA(problem_dict1, epoch, pop_size, pc, pm)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, problem, epoch=10000, pop_size=100, pc=0.95, pm=0.025, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.95
            pm (float): mutation probability, default = 0.025
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
        list_fitness = np.array([agent[self.ID_FIT][self.ID_TAR] for agent in self.pop])
        pop = []
        for i in range(0, self.pop_size):
            ### Selection
            # c1, c2 = self._get_parents_kway_tournament_selection__(pop, k_way=0.2)
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)

            w1 = self.pop[id_c1][self.ID_POS]
            w2 = self.pop[id_c2][self.ID_POS]
            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(w1, w2)

            ### Mutation, remove third loop here
            w1 = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, np.random.uniform(self.problem.lb, self.problem.ub), w1)
            w2 = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.pm, np.random.uniform(self.problem.lb, self.problem.ub), w2)

            if np.random.uniform() < 0.5:
                pop.append([w1, None])
            else:
                pop.append([w2, None])
        self.pop = self.update_fitness_population(pop)
