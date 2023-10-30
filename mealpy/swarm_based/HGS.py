#!/usr/bin/env python
# Created by "Thieu" at 15:37, 19/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalHGS(Optimizer):
    """
    The original version of: Hunger Games Search (HGS)

    Links:
        https://aliasgharheidari.com/HGS.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + PUP (float): [0.01, 0.2], The probability of updating position (L in the paper), default = 0.08
        + LH (float): [1000, 20000], Largest hunger / threshold, default = 10000

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HGS
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
    >>> model = HGS.OriginalHGS(epoch=1000, pop_size=50, PUP = 0.08, LH = 10000)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, Y., Chen, H., Heidari, A.A. and Gandomi, A.H., 2021. Hunger games search: Visions, conception, implementation,
    deep analysis, perspectives, and towards performance shifts. Expert Systems with Applications, 177, p.114864.
    """

    ID_HUN = 2  # ID for Hunger value

    def __init__(self, epoch: int = 10000, pop_size: int = 100, PUP: float = 0.08, LH: float = 10000, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            PUP (float): The probability of updating position (L in the paper), default = 0.08
            LH (float): Largest hunger / threshold, default = 10000
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.PUP = self.validator.check_float("PUP", PUP, (0, 1.0))
        self.LH = self.validator.check_float("LH", LH, [1, 20000])
        self.set_parameters(["epoch", "pop_size", "PUP", "LH"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        hunger = 1.0
        return Agent(solution=solution, hunger=hunger)

    def sech__(self, x):
        if np.abs(x) > 50:
            return 0.5
        return 2 / (np.exp(x) + np.exp(-x))

    def update_hunger_value__(self, pop=None, g_best=None, g_worst=None):
        # min_index = pop.index(min(pop, key=lambda x: x.target.fitness))
        # Eq (2.8) and (2.9)
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            # space: since we pass lower bound and upper bound as list. Better take the np.mean of them.
            space = np.mean(self.problem.ub - self.problem.lb)
            H = (pop[idx].target.fitness - g_best.target.fitness) / \
                (g_worst.target.fitness - g_best.target.fitness + self.EPSILON) * r * 2 * space
            if H < self.LH:
                H = self.LH * (1 + r)
            pop[idx].hunger += H

            if g_best.target.fitness == pop[idx].target.fitness:
                pop[idx].hunger = 0
        return pop

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Eq. (2.2)
        ### Find the current best and current worst
        _, (g_best, ), (g_worst, ) = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        pop = self.update_hunger_value__(self.pop, g_best, g_worst)

        ## Eq. (2.4)
        shrink = 2 * (1 - (epoch + 1) / self.epoch)
        total_hunger = np.sum([pop[idx].hunger for idx in range(0, self.pop_size)])

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            #### Variation control
            E = self.sech__(self.pop[idx].target.fitness - g_best.target.fitness)

            # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
            R = 2 * shrink * self.generator.random() - shrink  # Eq. (2.3)

            ## Calculate the hungry weight of each position
            if self.generator.random() < self.PUP:
                W1 = self.pop[idx].hunger * self.pop_size / (total_hunger + self.EPSILON) * self.generator.random()
            else:
                W1 = 1
            W2 = (1 - np.exp(-np.abs(self.pop[idx].hunger - total_hunger))) * self.generator.random() * 2

            ### Udpate position of individual Eq. (2.1)
            r1 = self.generator.random()
            r2 = self.generator.random()
            if r1 < self.PUP:
                pos_new = self.pop[idx].solution * (1 + self.generator.normal(0, 1))
            else:
                if r2 > E:
                    pos_new = W1 * g_best.solution + R * W2 * np.abs(g_best.solution - self.pop[idx].solution)
                else:
                    pos_new = W1 * g_best.solution - R * W2 * np.abs(g_best.solution - self.pop[idx].solution)
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
