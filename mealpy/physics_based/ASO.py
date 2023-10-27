#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalASO(Optimizer):
    """
    The original version of: Atom Search Optimization (ASO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2018.08.030
        2. https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): Depth weight, default = 10, depend on the problem
        + beta (float): Multiplier weight, default = 0.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ASO
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
    >>> model = ASO.OriginalASO(epoch=1000, pop_size=50, alpha = 50, beta = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2019. Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem. Knowledge-Based Systems, 163, pp.283-304.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, alpha: int = 10, beta: float = 0.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): [2, 20], Depth weight, default = 10
            beta (float): [0.1, 1.0], Multiplier weight, default = 0.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_int("alpha", alpha, [1, 100])
        self.beta = self.validator.check_float("beta", beta, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "alpha", "beta"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        mass = 0.0
        return Agent(solution=solution, velocity=velocity, mass=mass)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def update_mass__(self, population):
        list_fit = np.array([agent.target.fitness for agent in population])
        list_fit = np.exp(-(list_fit - np.max(list_fit)) / (np.max(list_fit) - np.min(list_fit) + self.EPSILON))
        list_fit = list_fit / np.sum(list_fit)
        for idx in range(0, self.pop_size):
            population[idx].mass = list_fit[idx]
        return population

    def find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin(iteration / self.epoch * np.pi / 2)
        rsmax = 1.24
        if radius / average_dist < rsmin:
            rs = rsmin
        else:
            if radius / average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
        return potential

    def acceleration__(self, population, g_best, iteration):
        eps = 2 ** (-52)
        pop = self.update_mass__(population)
        G = np.exp(-20.0 * iteration / self.epoch)
        k_best = int(self.pop_size - (self.pop_size - 2) * (iteration / self.epoch) ** 0.5) + 1
        if self.problem.minmax == "min":
            k_best_pop = sorted(pop, key=lambda agent: agent.mass, reverse=True)[:k_best].copy()
        else:
            k_best_pop = sorted(pop, key=lambda agent: agent.mass)[:k_best].copy()
        mk_average = np.mean([agent.solution for agent in k_best_pop])
        acc_list = np.zeros((self.pop_size, self.problem.n_dims))
        for idx in range(0, self.pop_size):
            dist_average = np.linalg.norm(pop[idx].solution - mk_average)
            temp = np.zeros((self.problem.n_dims))
            for atom in k_best_pop:
                # calculate LJ-potential
                radius = np.linalg.norm(pop[idx].solution - atom.solution)
                potential = self.find_LJ_potential__(iteration, dist_average, radius)
                temp += potential * self.generator.uniform(0, 1, self.problem.n_dims) * ((atom.solution - pop[idx].solution) / (radius + eps))
            temp = self.alpha * temp + self.beta * (g_best.solution - pop[idx].solution)
            # calculate acceleration
            acc = G * temp / pop[idx].mass
            acc_list[idx] = acc
        return acc_list

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate acceleration.
        atom_acc_list = self.acceleration__(self.pop, self.g_best, iteration=epoch)
        # Update velocity based on random dimensions and position of global best
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            velocity = self.generator.random(self.problem.n_dims) * self.pop[idx].velocity + atom_acc_list[idx]
            pos_new = self.pop[idx].solution + velocity
            # Relocate atom out of range
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        current_best = self.get_best_agent(pop_new, self.problem.minmax)
        if self.compare_target(self.g_best.target, current_best.target, self.problem.minmax):
            self.pop[self.generator.integers(0, self.pop_size)] = self.g_best.copy()
