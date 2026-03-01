#!/usr/bin/env python
# Created by "Thieu" at 11:40, 20/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBWO(Optimizer):
    """
    The original version of: Black Widow Optimization (BWO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103249

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): [0, 1], procreating rate, default = 0.6
        + cr (float): [0, 1], cannibalism rate, default = 0.44
        + pm (float): [0, 1], mutation rate, default = 0.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BWO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>>
    >>> model = BWO.OriginalBWO(epoch=1000, pop_size=50, pp=0.6, cr=0.44, pm=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hayyolalam, V. and Pourhaji Kazem, A.A., 2020. Black widow optimization algorithm: A novel meta-heuristic
    approach for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 87, 103249.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pp: float = 0.6, cr: float = 0.44,
                 pm: float = 0.4, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): procreating rate, default = 0.6
            cr (float): cannibalism rate, default = 0.44
            pm (float): mutation rate, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pp = self.validator.check_float("pp", pp, (0.0, 1.0))
        self.cr = self.validator.check_float("cr", cr, (0.0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pp", "cr", "pm"])
        self.sort_flag = False

    def initialize_variables(self):
        self.n_parents = max(2, int(self.pp * self.pop_size))
        if self.n_parents > self.pop_size:
            self.n_parents = self.pop_size
        self.n_mutate = max(0, int(self.pm * self.pop_size))

    def _procreate(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """
        Create two offspring from a pair of parents using blend crossover on Nvar/2 indices.
        """
        n_dims = self.problem.n_dims
        n_cross = max(1, n_dims // 2)
        idxs = self.generator.choice(n_dims, n_cross, replace=False)
        alpha = self.generator.random(len(idxs))
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[idxs] = alpha * parent1[idxs] + (1 - alpha) * parent2[idxs]
        child2[idxs] = alpha * parent2[idxs] + (1 - alpha) * parent1[idxs]
        return self.correct_solution(child1), self.correct_solution(child2)

    def _mutate(self, position: np.ndarray) -> np.ndarray:
        """
        Mutate one randomly selected position in the solution vector.
        """
        if self.problem.n_dims < 1:
            return position
        pos_new = position.copy()
        idx = self.generator.integers(0, self.problem.n_dims)
        pos_new[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
        return self.correct_solution(pos_new)

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pop1 = [agent.copy() for agent in pop_sorted[:self.n_parents]]

        pop2 = []
        for _ in range(self.n_parents):
            parent_idx = self.generator.choice(len(pop1), 2, replace=False)
            parent1, parent2 = pop1[parent_idx[0]], pop1[parent_idx[1]]
            female = self.get_better_agent(parent1, parent2, self.problem.minmax).copy()
            child1_pos, child2_pos = self._procreate(parent1.solution, parent2.solution)
            child1 = self.generate_empty_agent(child1_pos)
            child2 = self.generate_empty_agent(child2_pos)
            children = [child1, child2]
            if self.mode in self.AVAILABLE_MODES:
                self.update_target_for_population(children)
            else:
                for child in children:
                    child.target = self.get_target(child.solution)
            n_keep = self.generator.binomial(len(children), 1 - self.cr)
            if n_keep < 1:
                n_keep = 1
            children = self.get_sorted_population(children, self.problem.minmax)
            pop2.append(female)
            pop2.extend(children[:n_keep])

        pop3 = []
        if self.n_mutate > 0:
            for _ in range(self.n_mutate):
                parent = pop1[self.generator.integers(0, len(pop1))]
                pos_new = self._mutate(parent.solution)
                pop3.append(self.generate_empty_agent(pos_new))
            if self.mode in self.AVAILABLE_MODES:
                self.update_target_for_population(pop3)
            else:
                for agent in pop3:
                    agent.target = self.get_target(agent.solution)

        pop_new = pop2 + pop3
        if len(pop_new) < self.pop_size:
            needed = self.pop_size - len(pop_new)
            pop_new.extend([agent.copy() for agent in pop_sorted[:needed]])
        self.pop = self.get_sorted_and_trimmed_population(pop_new, self.pop_size, self.problem.minmax)
