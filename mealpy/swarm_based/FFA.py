#!/usr/bin/env python
# Created by "Thieu" at 17:13, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
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
    >>> from mealpy import FloatVar, FFA
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
    >>> model = FFA.OriginalFFA(epoch=1000, pop_size=50, gamma = 0.001, beta_base = 2, alpha = 0.2, alpha_damp = 0.99, delta = 0.05, exponent = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gandomi, A.H., Yang, X.S. and Alavi, A.H., 2011. Mixed variable structural optimization
    using firefly algorithm. Computers & Structures, 89(23-24), pp.2325-2336.
    [2] Arora, S. and Singh, S., 2013. The firefly optimization algorithm: convergence analysis and
    parameter selection. International Journal of Computer Applications, 69(3).
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, gamma: float = 0.001, beta_base: float = 2,
                 alpha: float = 0.2, alpha_damp: float = 0.99, delta: float = 0.05, exponent: int = 2, **kwargs: object) -> None:
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
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.gamma = self.validator.check_float("gamma", gamma, (0, 1.0))
        self.beta_base = self.validator.check_float("beta_base", beta_base, (0, 3.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.alpha_damp = self.validator.check_float("alpha_damp", alpha_damp, (0, 1.0))
        self.delta = self.validator.check_float("delta", delta, (0, 1.0))
        self.exponent = self.validator.check_int("exponent", exponent, [2, 4])
        self.set_parameters(["epoch", "pop_size", "gamma", "beta_base", "alpha", "alpha_damp", "delta", "exponent"])
        self.is_parallelizable = False
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
            agent = self.pop[idx].copy()
            pop_child = []
            for j in range(idx + 1, self.pop_size):
                # Move Towards Better Solutions
                if self.compare_target(self.pop[j].target, agent.target, self.problem.minmax):
                    # Calculate Radius and Attraction Level
                    rij = np.linalg.norm(agent.solution - self.pop[j].solution) / dmax
                    beta = self.beta_base * np.exp(-self.gamma * rij ** self.exponent)
                    # Mutation Vector
                    mutation_vector = self.delta * self.generator.uniform(0, 1, self.problem.n_dims)
                    temp = np.matmul((self.pop[j].solution - agent.solution), self.generator.uniform(0, 1, (self.problem.n_dims, self.problem.n_dims)))
                    pos_new = agent.solution + self.dyn_alpha * mutation_vector + beta * temp
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    pop_child.append(agent)
            if len(pop_child) < self.pop_size:
                pop_child += self.generate_population(self.pop_size - len(pop_child))
            local_best = self.get_best_agent(pop_child, self.problem.minmax)
            # Compare to Previous Solution
            if self.compare_target(local_best.target, agent.target, self.problem.minmax):
                self.pop[idx] = local_best
        self.pop.append(self.g_best)
        self.dyn_alpha = self.alpha_damp * self.alpha
