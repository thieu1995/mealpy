#!/usr/bin/env python
# Created by "Thieu" at 18:41, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEHO(Optimizer):
    """
    The original version of: Elephant Herding Optimization (EHO)

    Links:
        1. https://doi.org/10.1109/ISCBI.2015.8

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.3, 0.8], a factor that determines the influence of the best in each clan, default=0.5
        + beta (float): [0.3, 0.8], a factor that determines the influence of the x_center, default=0.5
        + n_clans (int): [3, 10], the number of clans, default=5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EHO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EHO.OriginalEHO(epoch=1000, pop_size=50, alpha = 0.5, beta = 0.5, n_clans = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2015, December. Elephant herding optimization.
    In 2015 3rd international symposium on computational and business intelligence (ISCBI) (pp. 1-5). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: float = 0.5, beta: float = 0.5, n_clans: int = 5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): a factor that determines the influence of the best in each clan, default=0.5
            beta (float): a factor that determines the influence of the x_center, default=0.5
            n_clans (int): the number of clans, default=5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 3.0))
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.n_clans = self.validator.check_int("n_clans", n_clans, [2, int(self.pop_size/5)])
        self.set_parameters(["epoch", "pop_size", "alpha", "beta", "n_clans"])
        self.n_individuals = int(self.pop_size / self.n_clans)
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.n_clans, self.n_individuals)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Clan updating operator
        pop_new = []
        for idx in range(0, self.pop_size):
            clan_idx = int(idx / self.n_individuals)
            pos_clan_idx = int(idx % self.n_individuals)
            if pos_clan_idx == 0:  # The best in clan, because all clans are sorted based on fitness
                center = np.mean(np.array([agent.solution for agent in self.pop_group[clan_idx]]), axis=0)
                pos_new = self.beta * center
            else:
                pos_new = self.pop_group[clan_idx][pos_clan_idx].solution + self.alpha * np.random.uniform() * \
                          (self.pop_group[clan_idx][0].solution - self.pop_group[clan_idx][pos_clan_idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        self.pop_group = self.generate_group_population(self.pop, self.n_clans, self.n_individuals)
        # Separating operator
        for idx in range(0, self.n_clans):
            self.pop_group[idx] = self.get_sorted_population(self.pop_group[idx], self.problem.minmax)
            self.pop_group[idx][-1] = self.generate_agent()
        self.pop = [agent for pack in self.pop_group for agent in pack]
