#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHHO(Optimizer):
    """
    The original version of: Harris Hawks Optimization (HHO)

    Links:
        1. https://doi.org/10.1016/j.future.2019.02.028

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HHO
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
    >>> model = HHO.OriginalHHO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H., 2019.
    Harris hawks optimization: Algorithm and applications. Future generation computer systems, 97, pp.849-872.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # -1 < E0 < 1
            E0 = 2 * self.generator.uniform() - 1
            # factor to show the decreasing energy of rabbit
            E = 2 * E0 * (1. - epoch * 1.0 / self.epoch)
            J = 2 * (1 - self.generator.uniform())

            # -------- Exploration phase Eq. (1) in paper -------------------
            if np.abs(E) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                if self.generator.random() >= 0.5:  # perch based on other family members
                    X_rand = self.pop[self.generator.integers(0, self.pop_size)].solution.copy()
                    pos_new = X_rand - self.generator.uniform() * np.abs(X_rand - 2 * self.generator.uniform() * self.pop[idx].solution)
                else:  # perch on a random tall tree (random site inside group's home range)
                    X_m = np.mean([x.solution for x in self.pop])
                    pos_new = (self.g_best.solution - X_m) - self.generator.uniform() * \
                              (self.problem.lb + self.generator.uniform() * (self.problem.ub - self.problem.lb))
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
            # -------- Exploitation phase -------------------
            else:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks
                if (self.generator.random() >= 0.5):
                    delta_X = self.g_best.solution - self.pop[idx].solution
                    if np.abs(E) >= 0.5:  # Hard besiege Eq. (6) in paper
                        pos_new = delta_X - E * np.abs(J * self.g_best.solution - self.pop[idx].solution)
                    else:  # Soft besiege Eq. (4) in paper
                        pos_new = self.g_best.solution - E * np.abs(delta_X)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_empty_agent(pos_new)
                    pop_new.append(agent)
                else:
                    LF_D = self.get_levy_flight_step(beta=1.5, multiplier=0.01, case=-1)
                    if np.abs(E) >= 0.5:  # Soft besiege Eq. (10) in paper
                        Y = self.g_best.solution - E * np.abs(J * self.g_best.solution - self.pop[idx].solution)
                    else:  # Hard besiege Eq. (11) in paper
                        X_m = np.mean([x.solution for x in self.pop])
                        Y = self.g_best.solution - E * np.abs(J * self.g_best.solution - X_m)
                    pos_Y = self.correct_solution(Y)
                    target_Y = self.get_target(pos_Y)
                    Z = Y + self.generator.uniform(self.problem.lb, self.problem.ub) * LF_D
                    pos_Z = self.correct_solution(Z)
                    target_Z = self.get_target(pos_Z)
                    if self.compare_target(target_Y, self.pop[idx].target, self.problem.minmax):
                        agent = self.generate_empty_agent(pos_Y)
                        agent.target = target_Y
                        pop_new.append(agent)
                        continue
                    if self.compare_target(target_Z, self.pop[idx].target, self.problem.minmax):
                        agent = self.generate_empty_agent(pos_Z)
                        agent.target = target_Z
                        pop_new.append(agent)
                        continue
                    pop_new.append(self.pop[idx].copy())
        if self.mode not in self.AVAILABLE_MODES:
            for idx, agent in enumerate(pop_new):
                pop_new[idx].target = self.get_target(agent.solution)
        else:
            pop_new = self.update_target_for_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
