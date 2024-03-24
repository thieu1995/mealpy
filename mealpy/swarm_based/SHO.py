#!/usr/bin/env python
# Created by "Thieu" at 10:55, 02/12/2019 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalSHO(Optimizer):
    """
    The original version of: Spotted Hyena Optimizer (SHO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.05.014

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
        + n_trials (int): default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SHO
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
    >>> model = SHO.OriginalSHO(epoch=1000, pop_size=50, h_factor = 5.0, n_trials = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
    technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, h_factor: float = 5., n_trials: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h_factor (float): default = 5, coefficient linearly decreased from 5.0 to 0
            n_trials (int): default = 10,
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.h_factor = self.validator.check_float("h_factor", h_factor, (0.5, 10.0))
        self.n_trials = self.validator.check_int("n_trials", n_trials, (1, float("inf")))
        self.set_parameters(["epoch", "pop_size", "h_factor", "n_trials"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            hh = self.h_factor - epoch * (self.h_factor / self.epoch)
            rd1 = self.generator.uniform(0, 1, self.problem.n_dims)
            rd2 = self.generator.uniform(0, 1, self.problem.n_dims)
            B = 2 * rd1
            E = 2 * hh * rd2 - hh

            if self.generator.random() < 0.5:
                D_h = np.abs(np.dot(B, self.g_best.solution) - self.pop[idx].solution)
                pos_new = self.g_best.solution - np.dot(E, D_h)
            else:
                N = 1
                for _ in range(0, self.n_trials):
                    pos_temp = self.g_best.solution + self.generator.normal(0, 1, self.problem.n_dims) * \
                              self.generator.uniform(self.problem.lb, self.problem.ub)
                    pos_new = self.correct_solution(pos_temp)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.g_best.target, self.problem.minmax):
                        N += 1
                        break
                    N += 1
                circle_list = []
                idx_list = self.generator.choice(range(0, self.pop_size), N, replace=False)
                for j in range(0, N):
                    D_h = np.abs(np.dot(B, self.g_best.solution) - self.pop[idx_list[j]].solution)
                    p_k = self.g_best.solution - np.dot(E, D_h)
                    circle_list.append(p_k)
                pos_new = np.mean(np.array(circle_list), axis=0)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(self.pop[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
