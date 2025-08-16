#!/usr/bin/env python
# Created by "Thieu" at 10:01, 16/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFDO(Optimizer):
    """
    The original version of: Fitness Dependent Optimizer (FDO)

    Notes:
        + https://doi.org/10.1109/ACCESS.2019.2907012
        + Inspired by the bee swarming reproductive process, this algorithm optimizes solutions based on their fitness values.
        + This algorithm mainly relies on Lévy flight techniques. Thanks to this method of generating random numbers
        according to the Lévy distribution, it is able to converge. However, in the design of the fitness weight
        condition, it is almost impossible for an update to occur when the fitness weight equals 1. This is the main drawback.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FDO
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
    >>> model = FDO.OriginalFDO(epoch=1000, pop_size=50, weight_factor=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdullah, J. M., & Ahmed, T. (2019).
    Fitness dependent optimizer: inspired by the bee swarming reproductive process. IEEe Access, 7, 43473-43486.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, weight_factor=0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            weight_factor (float): factor to adjust the fitness weight calculation, default = 0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.weight_factor = self.validator.check_float("weight_factor", weight_factor, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "weight_factor"])
        self.sort_flag = False
        self.is_parallelizable = False

    def before_main_loop(self):
        self.pop_pace = [0, ] * self.pop_size

    def get_fit_weight(self, best_fit, current_fit, weight_factor=0.1):
        """
        Calculate the fitness weight based on the best and current fitness values.

        Args:
            best_fit (float): The best fitness value found so far.
            current_fit (float): The current fitness value of the agent.
            weight_factor (float): A factor to adjust the weight calculation, default is 0.1.

        Returns:
            float: The fitness weight.
        """
        if best_fit == 0:
            return 0
        else:
            if self.problem.minmax == "min":
                if best_fit < (0.05 * current_fit):
                    return 0.2
                else:
                    return best_fit / current_fit - weight_factor
            else:
                if best_fit > (0.05 * current_fit):
                    return 0.2
                else:
                    return weight_factor - best_fit / current_fit

    def get_into_levy_bound(self, pos_new):
        """
        Ensure the new position is within the levy bounds.

        Args:
            pos_new (np.ndarray): The new position to be checked.

        Returns:
            np.ndarray: The position clipped to the problem bounds.
        """
        levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
        levy_up = self.problem.ub * np.abs(levy)
        levy_lb = self.problem.lb * np.abs(levy)
        pos_new = np.select(
            [pos_new > self.problem.ub, pos_new < self.problem.lb],
            [levy_up, levy_lb],
            default=pos_new
        )
        return pos_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update positions for each thief
        for idx in range(self.pop_size):
            fw = self.get_fit_weight(self.g_best.target.fitness, self.pop[idx].target.fitness, self.weight_factor)
            dist = self.g_best.solution - self.pop[idx].solution
            levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
            if fw == 1:
                pace = self.pop[idx].solution * levy
            elif fw == 0:
                pace = dist * levy
            else:
                pace = dist * fw * np.sign(levy)
            self.pop_pace[idx] = pace
            pos_new = self.pop[idx].solution + pace
            pos_new = self.get_into_levy_bound(pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Check if new position is better
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx] = agent
            else:
                # Alternative update strategy
                dist = self.g_best.solution - pos_new
                pos_new = pos_new + (dist * fw) + self.pop_pace[idx]
                pos_new = self.get_into_levy_bound(pos_new)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    self.pop[idx] = agent
                else:
                    # Third update strategy
                    levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
                    pos_new = self.pop[idx].solution + self.pop[idx].solution * levy
                    pos_new = self.get_into_levy_bound(pos_new)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                        self.pop[idx] = agent
