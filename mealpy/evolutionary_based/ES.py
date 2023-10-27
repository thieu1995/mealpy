#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalES(Optimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ES
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
    >>> model = ES.OriginalES(epoch=1000, pop_size=50, lamda = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.75, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "lamda"])
        self.n_child = int(self.lamda * self.pop_size)
        self.sort_flag = True
    
    def initialize_variables(self):
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance)
        return Agent(solution=solution, strategy=strategy)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(tau_p * self.generator.normal(0, 1.0, self.problem.n_dims) + tau * self.generator.normal(0, 1.0, self.problem.n_dims))
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        self.pop = self.get_sorted_and_trimmed_population(child + self.pop, self.pop_size, self.problem.minmax)


class LevyES(OriginalES):
    """
    The developed Levy-flight version: Evolution Strategies (ES)

    Notes:
        + The Levy-flight is applied, the flow and equations is changed
        + Link: https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ES
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
    >>> model = ES.LevyES(epoch=1000, pop_size=50, lamda = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, lamda: float = 0.75, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        super().__init__(epoch, pop_size, lamda, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(tau_p * self.generator.normal(0, 1.0, self.problem.n_dims) + tau * self.generator.normal(0, 1.0, self.problem.n_dims))
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        child_levy = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.get_levy_flight_step(multiplier=0.001, size=self.problem.n_dims, case=-1)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            stdevs = np.array([np.exp(tau_p * self.generator.normal(0, 1.0) + tau * self.generator.normal(0, 1.0)) for _ in range(self.problem.n_dims)])
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=stdevs)
            child_levy.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child_levy[-1].target = self.get_target(pos_new)
        child_levy = self.update_target_for_population(child_levy)
        self.pop = self.get_sorted_and_trimmed_population(child + child_levy + self.pop, self.pop_size, self.problem.minmax)


class CMA_ES(Optimizer):
    """
    The original version of: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

    Links:
        1. https://en.wikipedia.org/wiki/CMA-ES

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ES
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
    >>> model = ES.CMA_ES(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2), 159-195.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        step = self.generator.multivariate_normal(np.zeros(self.problem.n_dims), np.eye(self.problem.n_dims))
        return Agent(solution=solution, step=step)

    def before_main_loop(self):
        self.mu = int(np.round(self.pop_size / 2))
        self.ps = np.zeros(self.problem.n_dims)
        self.C = np.eye(self.problem.n_dims)
        self.pc = np.zeros(self.problem.n_dims)
        self.w = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size+1))
        self.w = self.w / np.sum(self.w)
        self.mu_eff = 1. / np.sum(self.w**2)      # Number of effective solutions
        # Step Size Control Parameters (c_sigma and d_sigma);
        sigma0 = 0.1 * (self.problem.ub - self.problem.lb)
        self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
        self.ds = 1 + self.cs + 2*np.max(np.sqrt((self.mu_eff - 1.)/(self.problem.n_dims + 1)) - 1, 0)
        self.ENN = np.sqrt(self.problem.n_dims) * (1 - 1.0/(4*self.problem.n_dims) + 1.0/(21*self.problem.n_dims**2))
        ## Covariance Update Parameters
        self.cc = (4+self.mu_eff/self.problem.n_dims) / (4 + self.problem.n_dims + 2 *self.mu_eff/self.problem.n_dims)
        self.c1 = 2. / ((self.problem.n_dims + 1.3)**2 + self.mu_eff)
        alpha_mu = 2
        self.cmu = min(1-self.c1, alpha_mu*(self.mu_eff-2+1/self.mu_eff)/((self.problem.n_dims+2)**2+alpha_mu*self.mu_eff/2))
        self.hth = (1.4 + 2 / (self.problem.n_dims + 1)) * self.ENN
        self.sigma = sigma0
        self.x_mean = np.mean([agent.solution for agent in self.pop[:self.mu]], axis=0)

    def update_step__(self, pop, cc):
        for idx in range(0, self.pop_size):
            pop[idx].step = self.generator.multivariate_normal(np.zeros(self.problem.n_dims), cc)
        return pop

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.x_mean + self.sigma * self.pop[idx].step
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_population(pop_new, self.problem.minmax)
        # Update MEan
        self.pop = self.update_step__(self.pop, self.C)
        self.x_step = np.zeros(self.problem.n_dims)
        for idx in range(0, self.mu):
            self.x_step += self.w[idx] * self.pop[idx].step
        self.x_mean = self.x_mean + self.sigma * self.x_step
        # Update Step Size
        t11 = np.dot(self.x_step, np.linalg.inv(np.linalg.cholesky(self.C).T))
        self.ps = (1 - self.cs)*self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * t11
        self.sigma = self.sigma * np.exp(self.cs / self.ds * (np.linalg.norm(self.ps)/self.ENN - 1))**0.3
        # Update Covariance Matrix
        if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * epoch)) < self.hth:
            hs = 1
        else:
            hs = 0
        delta = (1 - hs) * self.cc * (2 - self.cc)
        self.pc = (1 - self.cc)*self.pc + hs*np.sqrt(self.cc * (2 - self.cc)*self.mu_eff) * self.x_step
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc)) + delta * self.C
        for idx in range(0, self.mu):
            self.C = self.C + self.cmu * self.w[idx] * np.outer(self.pop[idx].step, self.pop[idx].step)
        # If Covariance Matrix is not Positive Defenite or Near Singular
        E, V = np.linalg.eig(self.C)
        E = np.diag(E)
        if np.any(np.diag(E) < 0):
            E[E < 0] = 0
            self.C = V * E / V


class Simple_CMA_ES(Optimizer):
    """
    The simple version of: Covariance Matrix Adaptation Evolution Strategy (Simple-CMA-ES)

    Links:
        1. Inspired from this version: https://github.com/jenkspt/CMA-ES
        2. https://ieeexplore.ieee.org/abstract/document/6790628/

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ES
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
    >>> model = ES.Simple_CMA_ES(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2), 159-195.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def before_main_loop(self):
        self.mu = int(np.round(self.pop_size / 2))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pos_list = np.array([agent.solution for agent in self.pop]).T
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pos_topk = np.array([agent.solution for agent in pop_sorted[:self.mu]]).T
        # Covariance of top k but using mean of entire population
        centered = pos_list - pos_topk.mean(1, keepdims=True)
        C = (centered @ centered.T) / (self.mu - 1)
        # Eigenvalue decomposition
        w, E = np.linalg.eigh(C)
        if np.any(np.diag(w) < 0):
            w[w < 0] = 0
        # Generate new population
        # Sample from multivariate gaussian with mean of topk
        N = self.generator.normal(size=(self.problem.n_dims, self.pop_size))
        X = pos_topk.mean(1, keepdims=True) + (E @ np.diag(np.sqrt(w)) @ N)
        X = X.T
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.correct_solution(X[idx])
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(pop_new[-1], self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
