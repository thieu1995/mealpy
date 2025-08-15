#!/usr/bin/env python
# Created by "Thieu" at 22:47, 15/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAFT(Optimizer):
    """
    The original version of: Ali baba and the Forty Thieves (AFT) optimizer

    Notes:
        + https://doi.org/10.1007/s00521-021-06392-x

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AFT
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
    >>> model = AFT.OriginalAFT(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Braik, M., Ryalat, M. H., & Al-Zoubi, H. (2022). A novel meta-heuristic algorithm for solving
    numerical optimization problems: Ali Baba and the forty thieves. Neural Computing and Applications, 34(1), 409-455.
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
        self.sort_flag = False

    def before_main_loop(self):
        # Initialize best positions (Marjaneh's astute plans)
        self.pop_best = self.pop.copy()     # It is like local best positions like in PSO
        # self.pop is population of alibaba ==> It will always update with new version no matter what

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate AFT parameters
        # Perception potential - decreases over iterations
        Pp = 0.1 * np.log(2.75 * (epoch/ self.epoch) ** 0.1)

        # Tracking distance - decreases over iterations
        Td = 2 * np.exp(-2 * (epoch / self.epoch) ** 2)

        # Generate random candidate followers indices
        random_followers = self.generator.integers(0, self.pop_size, size=self.pop_size)

        # Update positions for each thief
        for idx in range(self.pop_size):
            if self.generator.random() >= 0.5:
                # Thieves know where to search (TRUE case)
                if self.generator.random() > Pp:
                    # Case 1: Follow global best with tracking distance
                    direction = np.sign(self.generator.random() - 0.5)
                    movement = (Td * (self.pop_best[idx].solution - self.pop[idx].solution) * self.generator.random() +
                                Td * (self.pop[idx].solution - self.pop_best[random_followers[idx]].solution) * self.generator.random())
                    pos_new = self.g_best.solution + movement * direction
                else:
                    # Case 3: Random exploration within tracking distance
                    pos_new = self.problem.lb + Td * (self.problem.ub - self.problem.lb) * self.generator.random(self.problem.n_dims)
            else:
                # Thieves don't know where to search - opposite direction (Marjaneh's tricks)
                direction = np.sign(self.generator.random() - 0.5)
                movement = (Td * (self.pop_best[idx].solution - self.pop[idx].solution) * self.generator.random() +
                            Td * (self.pop[idx].solution - self.pop_best[random_followers[idx]].solution) * self.generator.random())
                pos_new = self.g_best.solution - movement * direction
            # Clip to bounds
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            self.pop[idx] = agent
            # self.pop_baba[idx] = agent
            if self.mode not in self.AVAILABLE_MODES:
                # self.pop_baba[idx].target = self.get_target(pos_new)
                self.pop[idx].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(self.pop)
            self.pop_best = self.greedy_selection_population(self.pop_best, self.pop, self.problem.minmax)
