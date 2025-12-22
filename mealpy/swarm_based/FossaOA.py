#!/usr/bin/env python
# Created by "Your Name" at 2024
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFossaOA(Optimizer):
    """
    The original version of: Fossa Optimization Algorithm (FossaOA)

    Links:
        1. https://doi.org/10.22266/ijies2024.1031.78

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FossaOA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = FossaOA.OriginalFossaOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hamadneh, T., Batiha, B., Dehghani, M., Werner, F., Montazeri, Z., Eguchi, K., & Bektemyssova, G. (2024). 
    Fossa Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm for Engineering Applications. 
    International Journal of Intelligent Engineering and Systems, 17(5).
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Create a list of fitness values for the current population
        # This is how we access fitness in Mealpy, not via a matrix variable
        pop_fitness = [agent.target.fitness for agent in self.pop]

        # --- Phase 1: Exploration (Attacking and moving towards the lemur) ---
        for idx in range(0, self.pop_size):
            # Find agents that are better than the current agent (Candidate Lemurs)
            # using self.compare_target handles both min and max problems correctly
            better_agents_indices = [i for i, fit in enumerate(pop_fitness) if self.compare_target(self.pop[i].target, self.pop[idx].target, self.problem.minmax)]

            if len(better_agents_indices) > 0:
                # Select a random lemur (SL)
                random_lemur_idx = self.generator.choice(better_agents_indices)
                selected_lemur = self.pop[random_lemur_idx]

                r = self.generator.random(self.problem.n_dims)
                I = self.generator.integers(1, 3, size=self.problem.n_dims)  # Random integer 1 or 2

                # Eq. 5: New Position Calculation
                pos_new = self.pop[idx].solution + r * (selected_lemur.solution - I * self.pop[idx].solution)
                
                # Check bounds and evaluate
                pos_new = self.correct_solution(pos_new)
                target = self.get_target(pos_new)

                # Greedy Selection (Eq. 6)
                if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                     self.pop[idx].update(solution=pos_new.copy(), target=target.copy())

        # --- Phase 2: Exploitation (Chasing to catch lemur) ---
        for idx in range(0, self.pop_size):
            r = self.generator.random(self.problem.n_dims)
            
            # Eq. 7: x_new = x + (1 - 2r) * (ub - lb)
            # Note: self.problem.ub/lb are numpy arrays provided by the Optimizer class
            step_size = (1 - 2 * r) * (self.problem.ub - self.problem.lb)
            pos_new = self.pop[idx].solution + step_size
            
            # Check bounds and evaluate
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)

            # Greedy Selection (Eq. 8)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())