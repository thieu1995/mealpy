#!/usr/bin/env python
# Created by "Thieu" at 23:58, 03/09/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalDOA(Optimizer):
    """
    The original version of: Dream Optimization Algorithm (DOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/178419-dream-optimization-algorithm-doa

    Notes:
        1. The Matlab code is sloppy and incorrect. Many variables are defined and computed but never actually used
        in the solution update process. For example, the variable fitness is calculated during the
        exploitation phase but not applied.

        2. The agent’s position is also not updated properly, meaning it remains unchanged even after
        the supposed update in the Matlab code.

        3. I suspect the results reported in this paper might not exist at all but were fabricated by
        the authors, since the benchmark functions are completely missing from the Matlab code.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DOA
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
    >>> model = DOA.OriginalDOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Lang, Y., & Gao, Y. (2025). Dream Optimization Algorithm (DOA): A novel metaheuristic optimization
    algorithm inspired by human dreams and its applications to real-world engineering problems.
    Computer Methods in Applied Mechanics and Engineering, 436, 117718.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Exploration phase (90% of iterations)
        exploration_end = int(9 * self.epoch / 10)
        if epoch <= exploration_end:
            # Divide into 5 groups
            pop_new = []
            for m in range(5):
                # Calculate k for current group
                aa = max(1, np.ceil(self.problem.n_dims / 8 / (m + 1)))
                bb = np.ceil(self.problem.n_dims / 3 / (m + 1)) + 1
                kk = self.generator.integers(aa, bb)
                # Group indices
                group_start = int((m / 5) * self.pop_size)
                group_end = int(((m + 1) / 5) * self.pop_size)
                # Update the best solution for current group
                pbest = self.get_best_agent(self.pop[group_start:group_end], self.problem.minmax)

                # Memory strategy and forgetting/supplementation
                for idx in range(group_start, group_end):
                    pos_new = pbest.solution.copy()
                    # Random permutation for dimensions to modify
                    in_indices = self.generator.choice(self.problem.n_dims, size=kk, replace=False)
                    if self.generator.random() < 0.9:
                        # Forgetting and supplementation strategy
                        cos_term = (np.cos((epoch + self.epoch / 10) * np.pi / self.epoch) + 1) / 2
                        for jdx in in_indices:
                            pos_new[jdx] = pbest.solution[jdx] + (self.generator.random() * (self.problem.ub[jdx] - self.problem.lb[jdx]) + self.problem.lb[jdx]) * cos_term
                            # Boundary handling
                            if pos_new[jdx] > self.problem.ub[jdx] or pos_new[jdx] < self.problem.lb[jdx]:
                                if self.problem.n_dims > 15:    # For high-dimensional problems
                                    rdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                                    pos_new[jdx] = self.pop[rdx].solution[jdx]
                                else:   # For low-dimensional problems
                                    pos_new[jdx] = self.generator.random() * (self.problem.ub[jdx] - self.problem.lb[jdx]) + self.problem.lb[jdx]
                    else:   # Alternative update strategy
                        for jdx in in_indices:
                            rdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                            pos_new[jdx] = self.pop[rdx].solution[jdx]
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                for idx in range(self.pop_size):
                    pop_new[idx].target = self.get_target(pop_new[idx].solution)
            pop_new = self.update_target_for_population(pop_new)
            self.pop = pop_new
        else:   # Exploitation phase (last 10% of iterations)
            # Update population
            pop_new = []
            for idx in range(self.pop_size):
                km = max(2, int(np.ceil(self.problem.n_dims / 3)))
                k = self.generator.integers(2, km + 1)
                in_indices = self.generator.choice(self.problem.n_dims, size=k, replace=False)
                pos_new = self.g_best.solution.copy()
                for jdx in in_indices:
                    cos_term = (np.cos(epoch * np.pi / self.epoch) + 1) / 2
                    pos_new[jdx] = pos_new[jdx] + (self.generator.random() * (self.problem.ub[idx] - self.problem.lb[idx]) + self.problem.lb[idx]) * cos_term
                    # Boundary handling
                    if pos_new[jdx] > self.problem.ub[jdx] or pos_new[jdx] < self.problem.lb[jdx]:
                        if self.problem.n_dims > 15:
                            rdx = self.generator.choice(list(set(range(self.pop_size)) - {idx}))
                            pos_new[jdx] = self.pop[rdx].solution[jdx]
                        else:
                            pos_new[jdx] = self.generator.random() * (self.problem.ub[jdx] - self.problem.lb[jdx]) + self.problem.lb[jdx]
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                for idx in range(self.pop_size):
                    pop_new[idx].target = self.get_target(pop_new[idx].solution)
            pop_new = self.update_target_for_population(pop_new)
            self.pop = pop_new
