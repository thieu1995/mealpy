#!/usr/bin/env python
# Created by "plendroik" at 21:05, 23/12/2025 ----------%
#       Email: yigit.temel61@gmail.com              %
# --------------------------------------------------%

import numpy as np
from math import pi, exp, sqrt
from mealpy.optimizer import Optimizer


class OriginalCrayOA(Optimizer):
    """
    The original version of: Crayfish Optimization Algorithm (COA)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-023-08844-x

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 10000
        + pop_size (int): number of population size, default = 100

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, COA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-100.,) * 30, ub=(100.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = COA.OriginalCOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Jia, H., Rao, H., Wen, C., & Mirjalili, S. (2023). 
    Crayfish optimization algorithm. Neural Computing and Applications, 1-43.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def p_obj(self, x):
        """
        Calculates probability based on temperature (Eq. 4 in the paper)
        """
        return 0.2 * (1 / (sqrt(2 * pi) * 3)) * exp(-(x - 25)**2 / (2 * 3**2))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq. (7): C decreases linearly from 2 to 1 over iterations
        C = 2 - (epoch / self.epoch)
        
        # Eq. (3): Temperature calculation
        temp = self.generator.random() * 15 + 20
        
        # Define food source (xf)
        # Improvement: Using average of current best and global best to avoid stagnation
        current_best = self.get_best_agent(self.pop, self.problem.minmax)
        xf = (current_best.solution + self.g_best.solution) / 2
        
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_current = self.pop[idx].solution
            pos_new = pos_current.copy()

            # --- Summer Resort Stage (Exploration) ---
            if temp > 30:
                if self.generator.random() < 0.5:
                    # Eq. (6)
                    rand_vec = self.generator.random(self.problem.n_dims)
                    pos_new = pos_current + C * rand_vec * (xf - pos_current)
                else:
                    # Eq. (8) and Eq. (9) - Competition stage
                    # Vectorized approach for efficiency
                    z_indices = self.generator.integers(0, self.pop_size, self.problem.n_dims)
                    z_solutions = np.array([self.pop[z_idx].solution[dim_idx] for dim_idx, z_idx in enumerate(z_indices)])
                    pos_new = pos_current - z_solutions + xf
            
            # --- Foraging Stage (Exploitation) ---
            else:
                Xfood = self.g_best.solution.copy()
                
                # Fitness ratio calculation (with stability for negative/zero values)
                current_fit = self.pop[idx].target.fitness
                best_fit = self.g_best.target.fitness
                EPSILON = 1e-16
                
                numer = abs(current_fit)
                denom = abs(best_fit) + EPSILON
                
                # Eq. (4) related P
                P = 3 * self.generator.random() * (numer / denom)
                p_val = self.p_obj(temp)
                
                if P > 2: # The food is too big
                    # Eq. (12)
                    Xfood = np.exp(-1/P) * Xfood
                    
                    # Eq. (13) - Vectorized
                    rand1 = self.generator.random(self.problem.n_dims)
                    rand2 = self.generator.random(self.problem.n_dims)
                    pos_new = pos_current + np.cos(2 * pi * rand1) * Xfood * p_val - np.sin(2 * pi * rand2) * Xfood * p_val
                else:
                    # Eq. (14)
                    rand_vec = self.generator.random(self.problem.n_dims)
                    pos_new = (pos_current - Xfood) * p_val + p_val * rand_vec * pos_current

            # Check bounds and generate agent
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            
            # Greedy Selection
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent