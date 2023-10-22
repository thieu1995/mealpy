#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEOA(Optimizer):
    """
    The developed version: Earthworm Optimisation Algorithm (EOA)

    Links:
        1. http://doi.org/10.1504/IJBIC.2015.10004283
        2. https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa

    Notes:
        The original version from matlab code above will not work well, even with small dimensions.
        I change updating process, change cauchy process using x_mean, use global best solution

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_c (float): (0, 1) -> better [0.5, 0.95], crossover probability
        + p_m (float): (0, 1) -> better [0.01, 0.2], initial mutation probability
        + n_best (int): (2, pop_size/2) -> better [2, 5], how many of the best earthworm to keep from one generation to the next
        + alpha (float): (0, 1) -> better [0.8, 0.99], similarity factor
        + beta (float): (0, 1) -> better [0.8, 1.0], the initial proportional factor
        + gama (float): (0, 1) -> better [0.8, 0.99], a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EOA
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
    >>> model = EOA.OriginalEOA(epoch=1000, pop_size=50, p_c = 0.9, p_m = 0.01, n_best = 2, alpha = 0.98, beta = 0.9, gama = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_c: float = 0.9, p_m: float = 0.01, n_best: int = 2,
                 alpha: float = 0.98, beta: float = 0.9, gama: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_c (float): default = 0.9, crossover probability
            p_m (float): default = 0.01 initial mutation probability
            n_best (int): default = 2, how many of the best earthworm to keep from one generation to the next
            alpha (float): default = 0.98, similarity factor
            beta (float): default = 0.9, the initial proportional factor
            gama (float): default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_c = self.validator.check_float("p_c", p_c, (0, 1.0))
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.n_best = self.validator.check_int("n_best", n_best, [2, int(self.pop_size / 2)])
        self.alpha = self.validator.check_float("alpha", alpha, (0, 1.0))
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.gama = self.validator.check_float("gama", gama, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "p_c", "p_m", "n_best", "alpha", "beta", "gama"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_beta = self.beta

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Update the pop best
        pop_elites, _, _ = self.get_special_agents(self.pop, n_best=1, minmax=self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            ### Reproduction 1: the first way of reproducing
            x_t1 = self.problem.lb + self.problem.ub - self.alpha * self.pop[idx].solution

            ### Reproduction 2: the second way of reproducing
            if idx >= self.n_best:  ### Select two parents to mate and create two children
                idx = int(self.pop_size * 0.2)
                if self.generator.uniform() < 0.5:  ## 80% parents selected from best population
                    idx1, idx2 = self.generator.choice(range(0, idx), 2, replace=False)
                else:  ## 20% left parents selected from worst population (make more diversity)
                    idx1, idx2 = self.generator.choice(range(idx, self.pop_size), 2, replace=False)
                r = self.generator.uniform()
                x_child = r * self.pop[idx2].solution + (1 - r) * self.pop[idx1].solution
            else:
                r1 = self.generator.integers(0, self.pop_size)
                x_child = self.pop[r1].solution
            x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)
        self.dyn_beta = self.gama * self.beta
        self.pop = self.get_sorted_and_trimmed_population(self.pop, self.pop_size, self.problem.minmax)

        pos_list = np.array([agent.solution for agent in self.pop])
        x_mean = np.mean(pos_list, axis=0)
        ## Cauchy mutation (CM)
        cauchy_w = self.g_best.solution.copy()
        pop_new = []
        for idx in range(self.n_best, self.pop_size):  # Don't allow the elites to be mutated
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            cauchy_w = np.where(condition, x_mean, cauchy_w)
            x_t1 = (cauchy_w + self.g_best.solution) / 2
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[self.n_best:] = self.greedy_selection_population(pop_new, self.pop[self.n_best:], self.problem.minmax)

        ## Elitism Strategy: Replace the worst with the previous generation's elites.
        self.pop, _, _ = self.get_special_agents(self.pop, minmax=self.problem.minmax)
        for idx in range(0, self.n_best):
            self.pop[self.pop_size - idx - 1] = pop_elites[idx].copy()

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, agent in enumerate(self.pop):
            if tuple(agent.solution.tolist()) in new_set:
                self.pop[idx] = self.generate_agent()
            else:
                new_set.add(tuple(agent.solution.tolist()))
