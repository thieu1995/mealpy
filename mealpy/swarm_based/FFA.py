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

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100
    + gamma (float): Light Absorption Coefficient, default = 0.001
    + beta_base (float): Attraction Coefficient Base Value, default = 2
    + alpha (float): Mutation Coefficient, default = 0.2
    + alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
    + delta (float): Mutation Step Size, default = 0.05
    + exponent (int): Exponent (m in the paper), default = 2

    References
    ----------
    .. [1] Gandomi, A.H., Yang, X.S. and Alavi, A.H., 2011. Mixed variable structural optimization
    using firefly algorithm. Computers & Structures, 89(23-24), pp.2325-2336.
    .. [2] Arora, S. and Singh, S., 2013. The firefly optimization algorithm: convergence analysis and
    parameter selection. International Journal of Computer Applications, 69(3).

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, FFA
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
    >>> model = FFA.OriginalFFA(epoch=1000, pop_size=50, gamma = 0.001, beta_base = 2, alpha = 0.2, alpha_damp = 0.99, delta = 0.05, exponent = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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


class MLFA_GD(Optimizer):
    """
    The original version of: Multiple Learning FA based on Gender Difference (MLFA-GD)

    Hyperparameters
    ---------------
    + epoch (int): Maximum number of iterations, default = 10000
    + pop_size (int): Population size, default = 100
    + m_females (int): Number of female fireflies selected by each male firefly, default = 3
    + beta0 (float): Base attractiveness at r=0., default=1.0
    + gama (float): Light absorption coefficient, default=1.0
    + alpha (float): Step size factor for randomization, default=0.2
    + k_rw (float): Number of chaotic random walks for the global best individual, default=10.

    Warnings
    --------
    This algorithm suffers from severe numerical instabilities:

    1. Distance Underflow (Eq. 3 & 12): The attractiveness term `exp(-gama * r^2)`
       evaluates to exactly 0.0 in large search bounds. Without distance normalization,
       attraction drops to zero, and the swarm paralyzes.
    2. Cauchy Mutation Explosion (Eq. 13): The female update utilizes an unscaled
       Cauchy distribution. Due to its heavy tails, it frequently generates massive
       values, throwing fireflies out of the search space boundaries.
    3. Formula Inconsistency (Eq. 8): The male update omits the random perturbation
       noise inherently needed in FA, risking premature stagnation in local optima.

    Note: Practical implementations must apply distance scaling and bounded mutations
    to make this algorithm functionally viable.

    References
    ----------
    .. [1] Zhang, Wenning, Chongyang Jiao, and Qinglei Zhou. "Firefly algorithm with multiple learning
    ability based on gender difference." Scientific Reports 15.1 (2025): 28400.
    https://doi.org/10.1038/s41598-025-09523-9

    Examples
    --------
    >>> import numpy as np
    >>> from mealpy import FloatVar, FFA
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
    >>> model = FFA.MLFA_GD(epoch=1000, pop_size=50, m_females=3, beta0=1.0, gama=1.0, alpha=0.2, k_rw=10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, m_females: int=3, beta0: float=1.0,
                 gama: float=1.0, alpha: float=0.2, k_rw: int=10, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Population size, default = 100
            m_females (int): Number of female fireflies selected by each male firefly.
            beta0 (float): Base attractiveness at r=0.
            gama (float): Light absorption coefficient.
            alpha (float): Step size factor for randomization.
            k_rw (float): Number of chaotic random walks for the global best individual.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 100000])
        self.m_females = self.validator.check_int("m_females", m_females, [1, pop_size//2])
        self.beta0 = self.validator.check_float("beta0", beta0, [0., 10.])
        self.gama = self.validator.check_float("gama", gama, [0., 10.])
        self.alpha = self.validator.check_float("alpha", alpha, [-10., 10.])
        self.k_rw = self.validator.check_int("k_rw", k_rw, [1, 1000])
        self.set_parameters(["epoch", "pop_size", "m_females", "beta0", "gama", "alpha", "k_rw"])
        self.sort_flag = False
        self.n_sub = pop_size // 2
        self.males = self.females = self.males_pbest = None

    def chaotic_map(self, k):
        """
        Generates k chaotic variants using the Logistic map.
        Maps the output strictly within the problem's search space.
        """
        ch = np.zeros((k, self.problem.n_dims))
        val = self.generator.random(self.problem.n_dims)
        for idx in range(k):
            ch[idx] = 4.0 * val * (1.0 - val)  # Logistic map equation
        return self.problem.lb + ch * (self.problem.ub - self.problem.lb)

    def evolve(self, epoch):
        """
        The main evolution step.
        """
        # Divide Population into Male and Female Subgroups
        males = self.pop[:self.n_sub].copy()
        males_pbest = males.copy()
        females = self.pop[self.n_sub:].copy()

        # --- MALE SUBGROUP UPDATE ---
        for idx in range(self.n_sub):
            idx_f = self.generator.choice(self.n_sub, self.m_females, replace=False)
            step = np.zeros(self.problem.n_dims)
            for jdx in idx_f:
                r = np.linalg.norm(males[idx].solution - females[jdx].solution)
                beta_k = self.beta0 * np.exp(-self.gama * (r ** 2))
                # Attraction or Escape
                if self.compare_target(females[jdx].target, males[idx].target, self.problem.minmax):
                    d_k = 1
                else:
                    d_k = -1
                lambda_k = self.generator.uniform(0, 1, self.problem.n_dims)
                step += d_k * beta_k * lambda_k * (females[jdx].solution - males[idx].solution)

            males[idx].solution = self.correct_solution(males[idx].solution + step)
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                males[idx].target = self.get_target(males[idx].solution)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            males = self.update_target_for_population(males)
        # Update Personal Historical Best for Males
        males_pbest = self.greedy_selection_population(males, males_pbest, self.problem.minmax)

        # --- FEMALE SUBGROUP UPDATE ---
        # Construct Generalized Centroid (y_GC) from male personal bests
        y_gc = np.mean([agent.solution for agent in males_pbest], axis=0)

        # Deep learning for y_GC using a randomly selected male
        rand_male = males[self.generator.integers(0, self.n_sub)].solution
        y_gc = y_gc + self.generator.standard_cauchy(self.problem.n_dims) * (rand_male - y_gc)
        y_gc_tar = self.get_target(self.correct_solution(y_gc))

        for idx in range(self.n_sub):
            if self.compare_target(y_gc_tar, females[idx].target, self.problem.minmax):
                r = np.linalg.norm(females[idx].solution - y_gc)
                pos_new = females[idx].solution + self.beta0 * np.exp(-self.gama * (r ** 2)) * (y_gc - females[idx].solution)
            else:
                pos_new = self.g_best.solution + self.generator.standard_cauchy(self.problem.n_dims)
            females[idx].solution = self.correct_solution(pos_new)
            # Update fitness in single mode
            if self.mode not in self.AVAILABLE_MODES:
                females[idx].target = self.get_target(females[idx].solution)
        # Update fitness in parallel modes
        if self.mode in self.AVAILABLE_MODES:
            females = self.update_target_for_population(females)

        ## Regroup the pop to match mealpy pipeline
        self.pop = males + females
        self.g_best = self.get_best_agent(self.pop + [self.g_best], self.problem.minmax, return_index=False)

        # --- RANDOM WALK FOR THE GLOBAL BEST ---
        eps = (self.epoch - epoch + 1) / self.epoch
        ch_vars = self.chaotic_map(self.k_rw)
        for kdx in range(self.k_rw):
            x_best_variant = (1 - eps) * self.g_best.solution + eps * ch_vars[kdx]
            pos_new = self.correct_solution(x_best_variant)
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.g_best.target, self.problem.minmax):
                self.g_best = agent
