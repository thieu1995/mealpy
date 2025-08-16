#!/usr/bin/env python
# Created by "Thieu" at 23:41, 15/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy.optimizer import Optimizer


class OriginalCDDO(Optimizer):
    """
    The original version of: Child Drawing Development Optimization (CCDO)

    Notes:
        + This source code was converted from the original Matlab implementation in the paper into Python.
        The Matlab code itself has many issues, for example, parameters are defined but never used.
        Several variables are declared, such as p1, p2, p3. Parameters like child skill rate and child level
        rate are initialized as hyperparameters at the beginning, but inside the loop they are randomly generated,
        which is inconsistent with the paper.

        + Moreover, the biggest flaw of this algorithm lies in the if–else condition during the update process.
        There is a high chance that neither condition will be executed, because the golden ratio is not necessarily
        within the interval [1.5, 2], as it is computed based on a random position. In addition, when comparing
        the position with a random integer T (hand pressure), it is unclear why this is done. It is highly likely
        that the algorithm will only execute that single condition.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CDDO
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
    >>> model = CDDO.OriginalCDDO(epoch=1000, pop_size=50, pattern_size=10, creativity_rate=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdulhameed, S., Rashid, T.A. Child Drawing Development Optimization Algorithm Based on
    Child’s Cognitive Development. Arab J Sci Eng 47, 1337–1351 (2022). https://doi.org/10.1007/s13369-021-05928-6
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, pattern_size=10,
                 creativity_rate=0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pattern_size (int): size of the pattern matrix, default = 10
            creativity_rate (float): creativity rate, default = 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pattern_size = self.validator.check_int("pattern_size", pattern_size, [1, 1000])
        self.creativity_rate = self.validator.check_float("creativity_rate", creativity_rate, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "pattern_size", "creativity_rate"])
        self.sort_flag = False

    def before_main_loop(self):
        self.LR = self.generator.uniform(0.1, 1.0)  # Child level rate
        self.SR = self.generator.uniform(0.1, 1.0)  # Child Skill Rate
        self.pop_local = self.pop.copy()
        # Golden ratio
        self.list_gr = []
        for idx in range(self.pop_size):
            p1 = self.generator.integers(0, self.problem.n_dims)
            p2 = self.generator.integers(0, self.problem.n_dims)
            if self.pop[idx].solution[p1] == 0:
                self.list_gr.append(self.pop[idx].solution[p2])
            else:
                self.list_gr.append(self.pop[idx].solution[p1] + self.pop[idx].solution[p2] / self.pop[idx].solution[p1])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Pattern matrix
        _, pattern, _ = self.get_special_agents(self.pop, n_best=self.pattern_size, minmax=self.problem.minmax)
        for idx in range(0, self.pop_size):
            hand_pressure = self.generator.integers(self.problem.lb[0], self.problem.ub[0] + 1)
            pp = self.generator.integers(0, self.problem.n_dims)
            pos_new = self.pop[idx].solution.copy()
            if self.pop[idx].solution[pp] <= hand_pressure:
                # Update the drawings
                pos_new = (self.list_gr[idx] + self.SR * self.generator.random(self.problem.n_dims) *
                           (self.pop_local[idx].solution - self.pop[idx].solution) +
                           self.LR * self.generator.random(self.problem.n_dims) *
                           (self.g_best.solution - self.pop[idx].solution))
                self.LR = self.generator.integers(6, 11) / 10
                self.SR = self.generator.integers(6, 11) / 10
            elif 1.5 < self.list_gr[idx] < 2:
                # Consider the learnt patterns
                pos_new = pattern[self.generator.integers(0, self.pattern_size)].solution - self.creativity_rate * self.pop_local[idx].solution
                self.LR = self.generator.integers(0, 6) / 10
                self.SR = self.generator.integers(0, 6) / 10
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            self.pop[idx] = agent
            if self.mode not in self.AVAILABLE_MODES:
                self.pop[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)
        # Update the local information
        self.pop_local = self.greedy_selection_population(self.pop_local, self.pop, self.problem.minmax)
