#!/usr/bin/env python
# Created by "Thieu" at 17:13, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalFFA(Optimizer):
    """
    The original version of: Firefly Algorithm (FFA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + gamma (float): Light Absorption Coefficient, default = 0.001
        + beta_base (float): Attraction Coefficient Base Value, default = 2
        + alpha (float): Mutation Coefficient, default = 0.2
        + alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
        + delta (float): Mutation Step Size, default = 0.05
        + exponent (int): Exponent (m in the paper), default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.FFA import OriginalFFA
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
    >>> gamma = 0.001
    >>> beta_base = 2
    >>> alpha = 0.2
    >>> alpha_damp = 0.99
    >>> delta = 0.05
    >>> exponent = 2
    >>> model = OriginalFFA(epoch, pop_size, gamma, beta_base, alpha, alpha_damp, delta, exponent)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Gandomi, A.H., Yang, X.S. and Alavi, A.H., 2011. Mixed variable structural optimization
    using firefly algorithm. Computers & Structures, 89(23-24), pp.2325-2336.
    [2] Arora, S. and Singh, S., 2013. The firefly optimization algorithm: convergence analysis and
    parameter selection. International Journal of Computer Applications, 69(3).
    """

    def __init__(self, epoch=10000, pop_size=100, gamma=0.001, beta_base=2, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
            delta (float): Mutation Step Size, default = 0.05
            exponent (int): Exponent (m in the paper), default = 2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.gamma = self.validator.check_float("gamma", gamma, (0, 1.0))
        self.beta_base = self.validator.check_float("beta_base", beta_base, (0, 3.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.alpha_damp = self.validator.check_float("alpha_damp", alpha_damp, (0, 1.0))
        self.delta = self.validator.check_float("delta", delta, (0, 1.0))
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.set_parameters(["epoch", "pop_size", "gamma", "beta_base", "alpha", "alpha_damp", "delta", "exponent"])
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_alpha = self.alpha  # Initial Value of Mutation Coefficient

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Maximum Distance
        dmax = np.sqrt(self.problem.n_dims)
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            pop_child = []
            for j in range(idx + 1, self.pop_size):
                # Move Towards Better Solutions
                if self.compare_agent(self.pop[j], agent):
                    # Calculate Radius and Attraction Level
                    rij = np.linalg.norm(agent[self.ID_POS] - self.pop[j][self.ID_POS]) / dmax
                    beta = self.beta_base * np.exp(-self.gamma * rij ** self.exponent)
                    # Mutation Vector
                    mutation_vector = self.delta * np.random.uniform(0, 1, self.problem.n_dims)
                    temp = np.matmul((self.pop[j][self.ID_POS] - agent[self.ID_POS]),
                                     np.random.uniform(0, 1, (self.problem.n_dims, self.problem.n_dims)))
                    pos_new = agent[self.ID_POS] + self.dyn_alpha * mutation_vector + beta * temp
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    target = self.get_target_wrapper(pos_new)
                    pop_child.append([pos_new, target])
            if len(pop_child) < self.pop_size:
                pop_child += self.create_population(self.pop_size - len(pop_child))
            _, local_best = self.get_global_best_solution(pop_child)
            # Compare to Previous Solution
            if self.compare_agent(local_best, agent):
                self.pop[idx] = local_best
        self.pop.append(self.g_best)
        self.dyn_alpha = self.alpha_damp * self.alpha
