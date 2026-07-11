#!/usr/bin/env python
# Created by "Ayoub Samir" on 05/01/2026
# Github: https://github.com/Ayoub-Samir
# -------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAAA(Optimizer):
    """
    The original version of: Artificial Algae Algorithm (AAA)

    Links:
        1. https://doi.org/10.1016/j.asoc.2015.03.003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + s_force (float): (-1000., 1000.0), shear force parameter (delta, in the paper); used as 2 in the paper's experiments
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

    def __init__(self, epoch: int = 10000, pop_size: int = 50, s_force: float = 2.0,
                 e_loss: float = 0.3, ap: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 50
            s_force: Shear force parameter (delta, in the paper)
            e_loss: Energy loss parameter
            ap: Adaptation parameter (Ap in the paper)
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.s_force = self.validator.check_float("s_force", s_force, (-1000., 1000.))
        self.e_loss = self.validator.check_float("e_loss", e_loss, (0., 1.0))
        self.ap = self.validator.check_float("ap", ap, (0., 1.0))
        self.set_parameters(["epoch", "pop_size", "s_force", "e_loss", "ap"])
        self.sort_flag = False
        self.is_parallelizable = False

    def before_main_loop(self) -> None:
        self.starvation = np.zeros(self.pop_size, dtype=int)
        fitness = np.array([agent.target.fitness for agent in self.pop])
        self.alg_size_matrix = self.calculate_greatness(np.ones(self.pop_size), fitness, self.problem.minmax)

    @staticmethod
    def calculate_greatness(greatness, fitness, minmax="min"):
        min_val = np.min(fitness)
        max_val = np.max(fitness)

        range_val = max_val - min_val
        if range_val == 0:
            norm_fitness = np.zeros_like(fitness)
        else:
            if minmax == "min":
                norm_fitness = (fitness - min_val) / range_val
            else:
                norm_fitness = (max_val - fitness) / range_val
        norm_fitness = 1 - norm_fitness

        # calculate greatness vector
        fks = np.abs(greatness / 2)
        m = norm_fitness / (fks + norm_fitness)
        dx = m * greatness
        greatness = greatness + dx
        return greatness

    @staticmethod
    def calculate_energy(greatness, minmax="min"):
        n = len(greatness)
        # 1. Find rank
        sorting = np.argsort(greatness)
        ranks = np.empty(n, dtype=int)
        ranks[sorting] = np.arange(n)
        # 2. Find fgreat_surface = rank^2
        fgreat_surface = (ranks.astype(float)) ** 2
        # 3. Min-Max
        min_val = np.min(fgreat_surface)
        max_val = np.max(fgreat_surface)

        range_val = max_val - min_val
        if range_val == 0:
            fgreat_surface = np.zeros_like(fgreat_surface)
        else:
            if minmax == "min":
                fgreat_surface = (fgreat_surface - min_val) / range_val
            else:
                fgreat_surface = (max_val - fgreat_surface) / range_val
        return fgreat_surface

    @staticmethod
    def calculate_friction(alg_size_matrix, minmax="min"):
        # Calculate radius r for all elements at once
        # r = ((alg_size * 3) / (4 * np.pi)) ** (1/3)
        r = ((alg_size_matrix * 3) / (4 * np.pi)) ** (1 / 3)
        # Calculate surface area for all elements at once
        fgreat_surface = 2 * np.pi * (r ** 2)
        # Perform Min-Max normalization using numpy vector operations
        max_val = np.max(fgreat_surface)
        min_val = np.min(fgreat_surface)
        # Avoid division by zero if max_val == min_val
        if max_val == min_val:
            return np.zeros_like(fgreat_surface)
        if minmax == "min":
            normalized_surface = (fgreat_surface - min_val) / (max_val - min_val)
        else:
            normalized_surface = (max_val - fgreat_surface) / (max_val - min_val)
        return normalized_surface

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch: The current iteration
        """
        energy = self.calculate_energy(self.alg_size_matrix, self.problem.minmax)
        alg_friction = self.calculate_friction(self.alg_size_matrix, self.problem.minmax)
        # 1. Helical Movement Phase
        for idx in range(self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            while energy[idx] >= 0:
                ndx = self.get_index_kway_tournament_selection(self.pop, k_way=0.2, output=1)[0]
                rd1, rd2, rd3 = self.generator.choice(self.problem.n_dims, 3, False)
                pos_new[rd1] += (self.pop[ndx].solution[rd1] - pos_new[rd1]) * (self.s_force - alg_friction[idx]) * ((self.generator.random() - 0.5) * 2)
                pos_new[rd2] += (self.pop[ndx].solution[rd2] - pos_new[rd2]) * (self.s_force - alg_friction[idx]) * np.cos(self.generator.random() * 2 * np.pi)
                pos_new[rd3] += (self.pop[ndx].solution[rd3] - pos_new[rd3]) * (self.s_force - alg_friction[idx]) * np.sin(self.generator.random() * 2 * np.pi)

                # Movement cost
                energy[idx] = energy[idx] - self.e_loss / 2
                pos_new = self.correct_solution(pos_new)
                agent_new = self.generate_agent(pos_new)
                if self.compare_target(agent_new.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = agent_new
                    self.starvation[idx] = 0
                else:
                    self.starvation[idx] += 1
                    energy[idx] = energy[idx] - self.e_loss / 2

        # 2. Reproduction: Replace smallest with biggest
        fitness = np.array([agent.target.fitness for agent in self.pop])
        self.alg_size_matrix = self.calculate_greatness(self.alg_size_matrix, fitness, self.problem.minmax)
        smallest = np.argmin(self.alg_size_matrix)
        biggest = np.argmax(self.alg_size_matrix)
        self.pop[smallest] = self.pop[biggest].copy()

        # 3. Adaptation Phase
        if self.generator.random() < self.ap:
            starving_idx = np.argmax(self.starvation)
            # Eq 13: Adapt toward the biggest colony
            pos_new = self.pop[starving_idx].solution + (self.g_best.solution - self.pop[starving_idx].solution) * self.generator.random(self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            self.pop[starving_idx] = agent
            self.starvation[starving_idx] = 0
