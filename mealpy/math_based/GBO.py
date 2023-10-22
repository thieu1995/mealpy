#!/usr/bin/env python
# Created by "Thieu" at 17:07, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGBO(Optimizer):
    """
    The original version of: Gradient-Based Optimizer (GBO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pr (float): [0.2, 0.8], Probability Parameter, default = 0.5
        + beta_min (float): Fixed parameter (no name in the paper), default = 0.2
        + beta_max (float): Fixed parameter (no name in the paper), default = 1.2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GBO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = GBO.OriginalGBO(epoch=1000, pop_size=50, pr = 0.5, beta_min = 0.2, beta_max = 1.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Bozorg-Haddad, O. and Chu, X., 2020. Gradient-based optimizer:
    A new metaheuristic optimization algorithm. Information Sciences, 540, pp.131-159.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pr: float = 0.5, beta_min: float = 0.2, beta_max: float = 1.2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pr (float): Probability Parameter, default = 0.5
            beta_min (float): Fixed parameter (no name in the paper), default = 0.2
            beta_max (float): Fixed parameter (no name in the paper), default = 1.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pr = self.validator.check_float("pr", pr, (0, 1.0))
        self.beta_min = self.validator.check_float("beta_min", beta_min, (0, 2.0))
        self.beta_max = self.validator.check_float("beta_max", beta_max, (0, 5.0))
        self.set_parameters(["epoch", "pop_size", "pr", "beta_min", "beta_max"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Eq.(14.2), Eq.(14.1)
        beta = self.beta_min + (self.beta_max - self.beta_min) * (1 - (epoch / self.epoch) ** 3) ** 2
        alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(beta * 3 * np.pi / 2)))

        pop_new = []
        for idx in range(0, self.pop_size):
            p1 = 2 * self.generator.random() * alpha - alpha
            p2 = 2 * self.generator.random() * alpha - alpha
            #  Four positions randomly selected from population
            r1, r2, r3, r4 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
            # Average of Four positions randomly selected from population
            r0 = (self.pop[r1].solution + self.pop[r2].solution + self.pop[r3].solution + self.pop[r4].solution) / 4
            # Randomization Epsilon
            epsilon = 5e-3 * self.generator.random()
            delta = 2 * self.generator.random() * np.abs(r0 - self.pop[idx].solution)
            step = (self.g_best.solution - self.pop[r1].solution + delta) / 2
            delta_x = self.generator.choice(range(0, self.pop_size)) * np.abs(step)
            x1 = self.pop[idx].solution - self.generator.normal() * p1 * 2 * delta_x * \
                 self.pop[idx].solution / (self.g_worst.solution - self.g_best.solution + epsilon) + \
                 self.generator.random() * p2 * (self.g_best.solution - self.pop[idx].solution)
            z = self.pop[idx].solution - self.generator.normal() * 2 * delta_x * \
                self.pop[idx].solution / (self.g_worst.solution - self.g_best.solution + epsilon)
            y_p = self.generator.random() * ((z + self.pop[idx].solution) / 2 + self.generator.random() * delta_x)
            y_q = self.generator.random() * ((z + self.pop[idx].solution) / 2 - self.generator.random() * delta_x)
            x2 = self.g_best.solution - self.generator.normal() * p1 * 2 * delta_x * self.pop[idx].solution / (y_p - y_q + epsilon) + \
                 self.generator.random() * p2 * (self.pop[r1].solution - self.pop[r2].solution)

            x3 = self.pop[idx].solution - p1 * (x2 - x1)
            ra = self.generator.random()
            rb = self.generator.random()
            pos_new = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3

            # Local escaping operator
            if self.generator.random() < self.pr:
                f1 = self.generator.uniform(-1, 1)
                f2 = self.generator.normal(0, 1)
                L1 = np.round(1 - self.generator.random())
                u1 = L1 * 2 * self.generator.random() + (1 - L1)
                u2 = L1 * self.generator.random() + (1 - L1)
                u3 = L1 * self.generator.random() + (1 - L1)
                L2 = np.round(1 - self.generator.random())
                x_rand = self.problem.generate_solution()
                x_p = self.pop[self.generator.choice(range(0, self.pop_size))].solution
                x_m = L2 * x_p + (1 - L2) * x_rand
                if self.generator.random() < 0.5:
                    pos_new = pos_new + f1 * (u1 * self.g_best.solution - u2 * x_m) + \
                              f2 * p1 * (u3 * (x2 - x1) + u2 * (self.pop[r1].solution - self.pop[r2].solution)) / 2
                else:
                    pos_new = self.g_best.solution + f1 * (u1 * self.g_best.solution - u2 * x_m) + f2 * p1 * \
                              (u3 * (x2 - x1) + u2 * (self.pop[r1].solution - self.pop[r2].solution)) / 2
            # Check if solutions go outside the search space and bring them back
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, best, worst = self.get_special_agents(self.pop, n_best=1, n_worst=1, minmax=self.problem.minmax)
        self.g_best, self.g_worst = best[0], worst[0]
