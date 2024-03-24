#!/usr/bin/env python
# Created by "Thieu" at 07:02, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import math
from mealpy.optimizer import Optimizer


class OriginalNRO(Optimizer):
    """
    The original version of: Nuclear Reaction Optimization (NRO)

    Links:
        1. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8720256

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MVO
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
    >>> model = MVO.DevMVO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wei, Z., Huang, C., Wang, X., Han, T. and Li, Y., 2019. Nuclear reaction optimization: A novel and
    powerful physics-based algorithm for global optimization. IEEE Access, 7, pp.66084-66109.
    [2] Wei, Z.L., Zhang, Z.R., Huang, C.Q., Han, B., Tang, S.Q. and Wang, L., 2019, June. An Approach
    Inspired from Nuclear Reaction Processes for Numerical Optimization. In Journal of Physics:
    Conference Series (Vol. 1213, No. 3, p. 032009). IOP Publishing.
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

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        xichma_v = 1
        xichma_u = ((math.gamma(1 + 1.5) * math.sin(math.pi * 1.5 / 2)) / (math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1.0 / 1.5)
        levy_b = (self.generator.normal(0, xichma_u ** 2)) / (np.sqrt(np.abs(self.generator.normal(0, xichma_v ** 2))) ** (1.0 / 1.5))
        # NFi phase
        Pb = self.generator.uniform()
        Pfi = self.generator.uniform()
        freq = 0.05
        alpha = 0.01
        pop_new = []
        for idx in range(self.pop_size):
            ## Calculate neutron vector Nei by Eq. (2)
            ## Random 1 more index to select neutron
            temp1 = list(set(range(0, self.pop_size)) - {idx})
            i1 = self.generator.choice(temp1, replace=False)
            Nei = (self.pop[idx].solution + self.pop[i1].solution) / 2
            ## Update population of fission products according to Eq.(3), (6) or (9);
            if self.generator.uniform() <= Pfi:
                ### Update based on Eq. 3
                if self.generator.uniform() <= Pb:
                    xichma1 = (np.log(epoch) * 1.0 / epoch) * np.abs(np.subtract(self.pop[idx].solution, self.g_best.solution))
                    gauss = np.array([self.generator.normal(self.g_best.solution[j], xichma1[j]) for j in range(self.problem.n_dims)])
                    Xi = gauss + self.generator.uniform() * self.g_best.solution - round(self.generator.random() + 1) * Nei
                ### Update based on Eq. 6
                else:
                    i2 = self.generator.choice(temp1, replace=False)
                    xichma2 = (np.log(epoch) * 1.0 / epoch) * np.abs(np.subtract(self.pop[i2].solution, self.g_best.solution))
                    gauss = np.array([self.generator.normal(self.pop[idx].solution[j], xichma2[j]) for j in range(self.problem.n_dims)])
                    Xi = gauss + self.generator.uniform() * self.g_best.solution - round(self.generator.random() + 2) * Nei
            ## Update based on Eq. 9
            else:
                i3 = self.generator.choice(temp1, replace=False)
                xichma2 = (np.log(epoch) * 1.0 / epoch) * np.abs(np.subtract(self.pop[i3].solution, self.g_best.solution))
                Xi = np.array([self.generator.normal(self.pop[idx].solution[j], xichma2[j]) for j in range(self.problem.n_dims)])
            ## Check the boundary and evaluate the fitness function
            pos_new = self.correct_solution(Xi)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # NFu phase
        ## Ionization stage
        ## Calculate the Pa through Eq. (10)
        pop_child = []
        ranked_pop = np.argsort([self.pop[i].target.fitness for i in range(self.pop_size)])
        for idx in range(self.pop_size):
            X_ion = self.pop[idx].solution.copy()
            if (ranked_pop[idx] * 1.0 / self.pop_size) < self.generator.random():
                i1, i2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                for j in range(self.problem.n_dims):
                    #### Levy flight strategy is described as Eq. 18
                    if self.pop[i2].solution[j] == self.pop[idx].solution[j]:
                        X_ion[j] = self.pop[idx].solution[j] + alpha * levy_b * (self.pop[idx].solution[j] - self.g_best.solution[j])
                    #### If not, based on Eq. 11, 12
                    else:
                        if self.generator.uniform() <= 0.5:
                            X_ion[j] = self.pop[i1].solution[j] + self.generator.uniform() * (self.pop[i2].solution[j] - self.pop[idx].solution[j])
                        else:
                            X_ion[j] = self.pop[i1].solution[j] - self.generator.uniform() * (self.pop[i2].solution[j] - self.pop[idx].solution[j])
            else:  #### Levy flight strategy is described as Eq. 21
                _, _, worst = self.get_special_agents(self.pop, n_worst=1, minmax=self.problem.minmax)
                X_worst = worst[0]
                for j in range(self.problem.n_dims):
                    ##### Based on Eq. 21
                    if X_worst.solution[j] == self.g_best.solution[j]:
                        X_ion[j] = self.pop[idx].solution[j] + alpha * levy_b * (self.problem.ub[j] - self.problem.lb[j])
                    ##### Based on Eq. 13
                    else:
                        X_ion[j] = self.pop[idx].solution[j] + round(self.generator.uniform()) * self.generator.uniform() * \
                                   (X_worst.solution[j] - self.g_best.solution[j])
            ## Check the boundary and evaluate the fitness function for X_ion
            pos_new = self.correct_solution(X_ion)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(self.pop, pop_child, self.problem.minmax)

        ## Fusion Stage
        ### all ions obtained from ionization are ranked based on (14) - Calculate the Pc through Eq. (14)
        pop_new = []
        ranked_pop = np.argsort([self.pop[i].target.fitness for i in range(self.pop_size)])
        for idx in range(self.pop_size):
            i1, i2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            #### Generate fusion nucleus
            if (ranked_pop[idx] * 1.0 / self.pop_size) < self.generator.random():
                t1 = self.generator.uniform() * (self.pop[i1].solution - self.g_best.solution)
                t2 = self.generator.uniform() * (self.pop[i2].solution - self.g_best.solution)
                temp2 = self.pop[i1].solution - self.pop[i2].solution
                X_fu = self.pop[idx].solution + t1 + t2 - np.exp(-np.linalg.norm(temp2)) * temp2
            #### Else
            else:
                ##### Based on Eq. 22
                check_equal = (self.pop[i1].solution == self.pop[i2].solution)
                if check_equal.all():
                    X_fu = self.pop[idx].solution + alpha * levy_b * (self.pop[idx].solution - self.g_best.solution)
                ##### Based on Eq. 16, 17
                else:
                    if self.generator.uniform() > 0.5:
                        X_fu = self.pop[idx].solution - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) *
                            (self.epoch - epoch) / self.epoch + 1) * (self.pop[i1].solution - self.pop[i2].solution)
                    else:
                        X_fu = self.pop[idx].solution - 0.5 * (np.sin(2 * np.pi * freq * epoch + np.pi) * epoch / self.epoch + 1) * \
                               (self.pop[i1].solution - self.pop[i2].solution)
            pos_new = self.correct_solution(X_fu)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
