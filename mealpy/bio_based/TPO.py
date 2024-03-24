#!/usr/bin/env python
# Created by "Thieu" at 00:27, 18/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevTPO(Optimizer):
    """
    The original version: Tree Physiology Optimization (TPO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/63982-tree-physiology-optimization-tpo-algorithm-for-stochastic-test-function-optimization

    Notes:
        1. The paper is difficult to read and understand, and the provided MATLAB code is also challenging to understand.
        2. Based on my idea:
            + pop_size = number of branhes, the population size should be equal to the number of branches.
            + The number of leaves should be calculated as int(sqrt(pop_size) + 1), so we don't need to specify the n_leafs parameter, which will also reduce computation time.
            + When using this algorithm, especially when setting stopping conditions, be careful and set it to the FE type.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [-10, 10.] -> better [0.2, 0.5], Absorption constant for tree root elongation, default = 0.5
        + beta (float): [-100, 100.] -> better [10, 50], Diversification facor of tree shoot, default=50.
        + theta (float): (0, 1.0] -> better [0.5, 0.9], Factor to reduce randomization, Theta = Power law to reduce randomization as iteration increases, default=0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TPO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = TPO.DevTPO(epoch=1000, pop_size=50, alpha = 0.3, beta = 50., theta = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Halim, A. H., & Ismail, I. (2017). Tree physiology optimization in benchmark function and
    traveling salesman problem. Journal of Intelligent Systems, 28(5), 849-871.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: float = 0.3, beta: float = 50.0, theta: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): Absorption constant for tree root elongation, default=0.3
            beta (float): Diversification factor of tree shoot, default=50.
            theta (float): Factor to reduce randomization, Theta = Power law to reduce randomization as iteration increases, default=0.9
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])     # Number of branches
        self.alpha = self.validator.check_float("alpha", alpha, [-10.0, 10.])
        self.beta = self.validator.check_float("beta", beta, [-100., 100])
        self.theta = self.validator.check_float("theta", theta, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta", "theta"])
        self.sort_flag = False

    def initialize_variables(self):
        """
        The idea is a tree has a pop_size of branches (n_branches), each branch will have several leafs.
        """
        self.n_leafs = int(np.sqrt(self.pop_size) + 1)  # Number of leafs
        self._theta = self.theta
        self.roots = self.generator.uniform(0, 1, (self.n_leafs, self.problem.n_dims))

    def initialization(self):
        self.pop_total = []
        self.pop = []                       # The best leaf in each branches
        for idx in range(self.pop_size):
            leafs = self.generate_population(self.n_leafs)
            best = self.get_best_agent(leafs, self.problem.minmax)
            self.pop.append(best)
            self.pop_total.append(leafs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            pos_list = np.array([agent.solution for agent in self.pop_total[idx]])
            carbon_gain = self._theta * self.g_best.solution - pos_list
            roots_old = np.copy(self.roots)
            self.roots += self.alpha * carbon_gain * self.generator.uniform(-0.5, 0.5, (self.n_leafs, self.problem.n_dims))
            nutrient_value = self._theta * (self.roots - roots_old)
            pos_list_new = self.g_best.solution + self.beta * nutrient_value
            pop_new = []
            for jdx in range(0, self.n_leafs):
                pos_new = self.correct_solution(pos_list_new[jdx])
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop_total[idx][jdx] = self.get_better_agent(agent, self.pop_total[idx][jdx], self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                self.pop_total[idx] = self.greedy_selection_population(pop_new, self.pop_total[idx], self.problem.minmax)
        self._theta = self._theta * self.theta
        for idx in range(0, self.pop_size):
            best = self.get_best_agent(self.pop_total[idx], self.problem.minmax)
            self.pop[idx] = best
