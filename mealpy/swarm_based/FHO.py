#!/usr/bin/env python
# Created by "Thieu" at 21:16, 26/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFHO(Optimizer):
    """
    The original version of: Fire Hawk Optimization (FHO)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.

    Note
    ~~~~
    1. There are discrepancies between the author's MATLAB code and the paper.
    2. This Python version strictly follows what is written in the paper.

    Links
    -----
    1. https://doi.org/10.1007/s10462-022-10173-w
    2. https://www.mathworks.com/matlabcentral/fileexchange/114325-fire-hawk-optimizer-fho-a-novel-metaheuristic-algorithm

    References
    ~~~~~~~~~~
    1. Azizi, M., Talatahari, S., & Gandomi, A. H. (2022). Fire Hawk Optimizer: a novel metaheuristic algorithm. Artificial Intelligence Review, 1-77.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FHO
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
    >>> model = FHO.OriginalFHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch: int):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Generate random integer for the number of Fire Hawks (n)
        n = self.generator.integers(1, self.pop_size // 5 + 1)
        m = self.pop_size - n  # Number of Preys (PR)

        # Sort candidates to determine Fire Hawks (best) and Preys (rest)
        _, sorted_indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
        pos_list = np.array([agent.solution for agent in self.pop])
        FH = pos_list[sorted_indices[:n]]
        PR = pos_list[sorted_indices[n:]]

        new_FH = np.zeros_like(FH)
        new_PR = np.zeros_like(PR)

        # Eq. 5: Calculate total distance between Fire Hawks and Preys
        # dist matrix shape: (n, m)
        dist = np.linalg.norm(FH[:, np.newaxis, :] - PR[np.newaxis, :, :], axis=2)

        # Determine territory by assigning preys to the nearest Fire Hawk
        territory_assignments = np.argmin(dist, axis=0)

        # Update Fire Hawks' positions
        for idx in range(n):
            r1, r2 = self.generator.random(2)
            # Select another Fire Hawk randomly
            available_fh = [jdx for jdx in range(n) if jdx != idx]
            jdx = self.generator.choice(available_fh) if available_fh else idx
            # Eq. 6: New position of Fire Hawks
            new_FH[idx] = FH[idx] + (r1 * self.g_best.solution - r2 * FH[jdx])
        # Calculate safe place outside all territories (Eq. 10)
        SP_global = np.mean(PR, axis=0) if m > 0 else np.zeros(self.problem.n_dims)

        # Update Preys' positions
        for idx in range(m):
            l = territory_assignments[idx]
            r3, r4, r5, r6 = self.generator.random(4)

            # Eq. 9: Calculate safe place under l-th Fire Hawk territory
            preys_in_territory = PR[territory_assignments == l]
            SP_l = np.mean(preys_in_territory, axis=0) if len(preys_in_territory) > 0 else PR[idx]

            # Eq. 7: Update position inside the territory
            PR_temp = PR[idx] + (r3 * FH[l] - r4 * SP_l)

            # Select an alternative Fire Hawk
            available_fh = [jdx for jdx in range(n) if jdx != l]
            jdx = np.random.choice(available_fh) if available_fh else l

            # Eq. 8: Update position outside the territory
            new_PR[idx] = PR_temp + (r5 * FH[jdx] - r6 * SP_global)

        # Merge new populations and enforce boundary constraints
        new_X = np.vstack((new_FH, new_PR))
        pop_new = []
        for idx in range(self.pop_size):
            pos_new = self.correct_solution(new_X[idx])
            pop_new.append(self.generate_empty_agent(pos_new))
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        # Evaluate fitness in parallel
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        # Update population
        self.pop = pop_new
