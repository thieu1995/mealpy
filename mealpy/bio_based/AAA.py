#!/usr/bin/env python
# Created by "Thieu" at 12:00, 21/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import math
import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.target import Target


class OriginalAAA(Optimizer):
    """
    The original version of: Artificial Algae Algorithm (AAA)

    Notes:
        + A bio-inspired metaheuristic based on the evolutionary process, adaptation, and helical movement of microalgae
        + Proposed for nonlinear global optimization

    Links:
        1. https://doi.org/10.1016/j.asoc.2015.03.003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + s_force (float): (0, 5.0), shear force parameter (delta, Δ, in the paper); used as 2 in the paper's experiments
        + e_loss (float): (0, 1.0), energy loss parameter (e in the paper); used as 0.3 in the paper's experiments
        + ap (float): (0, 1.0), adaptation parameter (Ap in the paper); used as 0.5 (benchmarks) or 1 (design problem)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AAA
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
    >>> model = AAA.OriginalAAA(epoch=1000, pop_size=50, s_force=2.0, e_loss=0.3, ap=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Uymaz, S. A., Tezel, G., and Yel, E. (2015). Artificial algae algorithm (AAA) for nonlinear global optimization.
    Applied Soft Computing, 31, 153-171.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 40, s_force: float = 2.0,
                 e_loss: float = 0.3, ap: float = 0.5, **kwargs: object) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 40
            s_force: Shear force parameter (delta, Δ, in the paper)
            e_loss: Energy loss parameter
            ap: Adaptation parameter (Ap in the paper)
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.s_force = self.validator.check_float("s_force", s_force, (0., 5.0))
        self.e_loss = self.validator.check_float("e_loss", e_loss, (0., 1.0))
        self.ap = self.validator.check_float("ap", ap, (0., 1.0))
        self.set_parameters(["epoch", "pop_size", "s_force", "e_loss", "ap"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self) -> None:
        self.starve_alg = np.zeros(self.pop_size, dtype=int)
        self.alg_size_matrix = np.ones(self.pop_size)
        self.convergence_curve = np.zeros(self.epoch)
        self.best_alg = None
        self.best_fit = None
        self.alg = None
        self.fitness_array = None
        self.t = 1
        self.nfe_counter = 0

    def initialization(self) -> None:
        if self.pop is None:
            lb = self.problem.lb
            ub = self.problem.ub
            dim = self.problem.n_dims
            alg = np.zeros((self.pop_size, dim))
            for idx in range(dim):
                alg[:, idx] = self.generator.uniform(0, 1, self.pop_size) * (ub[idx] - lb[idx]) + lb[idx]
            self.pop = []
            for idx in range(self.pop_size):
                agent = self.generate_empty_agent(alg[idx, :])
                agent.target = self.get_target(agent.solution)
                self.pop.append(agent)

    def before_main_loop(self) -> None:
        self.alg = np.array([agent.solution for agent in self.pop])
        self.fitness_array = np.array([agent.target.fitness for agent in self.pop])
        min_fit_index = np.argmin(self.fitness_array)
        self.best_alg = self.alg[min_fit_index, :].copy()
        self.best_fit = self.fitness_array[min_fit_index]
        self._calculate_greatness(self.alg_size_matrix, self.fitness_array.copy())
        if self.epoch > 0:
            self.convergence_curve[0] = self.best_fit
        self.t = 1

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        max_fe = self.epoch * self.pop_size
        if self.termination is not None and self.termination.max_fe is not None:
            max_fe = self.termination.max_fe

        lb = self.problem.lb
        ub = self.problem.ub
        dim = self.problem.n_dims

        energy = self._calculate_energy(self.alg_size_matrix.copy())
        alg_friction = self._calculate_friction(self.alg_size_matrix.copy())
        for idx in range(self.pop_size):
            istarve = 0
            while energy[idx] >= 0 and self.nfe_counter < max_fe:
                neighbor = self._tournament_selection(self.fitness_array.copy())
                while neighbor == idx:
                    neighbor = self._tournament_selection(self.fitness_array)

                dimension_1 = self.rng.randint(0, dim - 1)
                dimension_2 = self.rng.randint(0, dim - 1)
                dimension_3 = self.rng.randint(0, dim - 1)

                if dim == 2:
                    while dimension_1 == dimension_2:
                        dimension_2 = self.rng.randint(0, dim - 1)
                    new_alg = self.alg[idx, :].copy()
                    new_alg[dimension_1] = new_alg[dimension_1] + (
                        self.alg[neighbor, dimension_1] - new_alg[dimension_1]) * (self.s_force - alg_friction[idx]) * (
                        (self.rng.random() - 0.5) * 2)
                    new_alg[dimension_2] = new_alg[dimension_2] + (
                        self.alg[neighbor, dimension_2] - new_alg[dimension_2]) * (
                        self.s_force - alg_friction[idx]) * math.sin(self.rng.random() * math.tau)
                elif dim >= 3:
                    while (dimension_1 == dimension_2 or dimension_1 == dimension_3 or dimension_2 == dimension_3):
                        dimension_2 = self.rng.randint(0, dim - 1)
                        dimension_3 = self.rng.randint(0, dim - 1)

                    new_alg = self.alg[idx, :].copy()
                    new_alg[dimension_1] = new_alg[dimension_1] + (
                        self.alg[neighbor, dimension_1] - new_alg[dimension_1]) * (
                        self.s_force - alg_friction[idx]) * ((self.rng.random() - 0.5) * 2)
                    new_alg[dimension_2] = new_alg[dimension_2] + (
                        self.alg[neighbor, dimension_2] - new_alg[dimension_2]) * (
                        self.s_force - alg_friction[idx]) * math.cos(self.rng.random() * math.tau)
                    new_alg[dimension_3] = new_alg[dimension_3] + (
                        self.alg[neighbor, dimension_3] - new_alg[dimension_3]) * (
                        self.s_force - alg_friction[idx]) * math.sin(self.rng.random() * math.tau)

                new_alg = np.clip(new_alg, lb, ub)
                new_alg_fit = self.get_target(new_alg).fitness
                energy[idx] = energy[idx] - self.e_loss / 2

                if new_alg_fit < self.fitness_array[idx]:
                    self.alg[idx, :] = new_alg.copy()
                    self.fitness_array[idx] = new_alg_fit
                    istarve = 1
                else:
                    energy[idx] = energy[idx] - self.e_loss / 2

                value = min(self.fitness_array)
                index = np.argmin(self.fitness_array)
                if value < self.best_fit:
                    self.best_fit = value
                    self.best_alg = self.alg[index, :].copy()

                if self.nfe_counter % self.pop_size == 0:
                    if self.t < len(self.convergence_curve):
                        self.convergence_curve[self.t] = self.best_fit
                    self.t = self.t + 1

            if istarve == 0:
                self.starve_alg[idx] = self.starve_alg[idx] + 1

        if self.nfe_counter < max_fe:
            self._calculate_greatness(self.alg_size_matrix, self.fitness_array.copy())
            rand_dim = self.rng.randint(0, dim - 1)
            minindex = np.argmin(self.alg_size_matrix)
            maxindex = np.argmax(self.alg_size_matrix)
            self.alg[minindex, rand_dim] = self.alg[maxindex, rand_dim]

            index3 = np.argmax(self.starve_alg)
            if self.rng.random() < self.ap:
                for i in range(dim):
                    self.alg[index3, i] = self.alg[index3, i] + (self.best_alg[i] - self.alg[index3, i]) * self.rng.random()

        for idx in range(self.pop_size):
            self.pop[idx].update(solution=self.alg[idx, :], target=Target(objectives=self.fitness_array[idx]))

        if self.best_alg is not None:
            self.pop[0].update(solution=self.best_alg.copy(), target=Target(objectives=self.best_fit))

    @staticmethod
    def _calculate_greatness(greatness, fitness_array1):
        max_val = max(fitness_array1)
        min_val = min(fitness_array1)

        for i in range(len(fitness_array1)):
            fitness_array1[i] = (fitness_array1[i] - min_val) / (max_val - min_val)
            fitness_array1[i] = 1 - fitness_array1[i]

        for i in range(len(greatness)):
            fks = abs(greatness[i] / 2)
            m = fitness_array1[i] / (fks + fitness_array1[i])
            dx = m * greatness[i]
            greatness[i] = greatness[i] + dx

    @staticmethod
    def _calculate_energy(greatness):
        sorting = np.ones(len(greatness), "int")
        fgreat_surface = np.zeros(len(greatness))

        for i in range(0, len(greatness)):
            sorting[i] = i

        for i in range(len(greatness) - 1):
            for j in range(i + 1, len(greatness)):
                if greatness[sorting[i]] > greatness[sorting[j]]:
                    sorting[i], sorting[j] = sorting[j], sorting[i]
            fgreat_surface[sorting[i]] = i ** 2

        fgreat_surface[sorting[len(greatness) - 1]] = (i + 1) ** 2
        max_val = max(fgreat_surface)
        min_val = min(fgreat_surface)

        for i in range(len(fgreat_surface)):
            fgreat_surface[i] = (fgreat_surface[i] - min_val) / (max_val - min_val)

        return fgreat_surface

    @staticmethod
    def _calculate_friction(alg_size_matrix):
        fgreat_surface = np.zeros(len(alg_size_matrix))
        for i in range(len(alg_size_matrix)):
            r = ((alg_size_matrix[i] * 3) / (4 * math.pi)) ** (1 / 3)
            fgreat_surface[i] = 2 * math.pi * (r ** 2)

        max_val = max(fgreat_surface)
        min_val = min(fgreat_surface)

        for i in range(len(fgreat_surface)):
            fgreat_surface[i] = (fgreat_surface[i] - min_val) / (max_val - min_val)

        return fgreat_surface

    def _tournament_selection(self, fitness_array):
        individual1 = self.rng.randint(0, len(fitness_array) - 1)
        individual2 = self.rng.randint(0, len(fitness_array) - 1)

        while individual1 == individual2:
            individual2 = self.rng.randint(0, len(fitness_array) - 1)

        if fitness_array[individual1] < fitness_array[individual2]:
            return individual1
        return individual2
