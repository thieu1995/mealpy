#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalESOA(Optimizer):
    """
    The original version of: Egret Swarm Optimization Algorithm (ESOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/115595-egret-swarm-optimization-algorithm-esoa
        2. https://www.mdpi.com/2313-7673/7/4/144

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, ESOA
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
    >>> model = ESOA.OriginalESOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, Z., Francis, A., Li, S., Liao, B., Xiao, D., Ha, T. T., ... & Cao, X. (2022). Egret Swarm Optimization Algorithm:
    An Evolutionary Computation Approach for Model Free Optimization. Biomimetics, 7(4), 144.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        weights = self.generator.uniform(-1., 1., self.problem.n_dims)
        m = np.zeros(self.problem.n_dims)
        v = np.zeros(self.problem.n_dims)
        return Agent(solution=solution, weights=weights, local_solution=solution.copy(), m=m, v=v)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """
            ID_WEI = 2
            ID_LOC_X = 3
            ID_LOC_Y = 4
            ID_G = 5
            ID_M = 6
            ID_V = 7
        """
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        agent.g = (np.sum(agent.weights * agent.solution) - agent.target.fitness) * agent.solution
        return agent

    def initialize_variables(self):
        self.beta1 = 0.9
        self.beta2 = 0.99

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        hop = self.problem.ub - self.problem.lb
        for idx in range(0, self.pop_size):
            # Individual Direction
            p_d = self.pop[idx].local_solution - self.pop[idx].solution
            p_d = p_d * (self.pop[idx].local_target.fitness - self.pop[idx].target.fitness)
            p_d = p_d / ((np.sum(p_d) + self.EPSILON)**2)
            d_p = p_d + self.pop[idx].g

            # Group Direction
            c_d = self.g_best.solution - self.pop[idx].solution
            c_d = c_d * (self.g_best.target.fitness - self.pop[idx].target.fitness)
            c_d = c_d / ((np.sum(c_d) + self.EPSILON)**2)
            d_g = c_d + self.g_best.g

            # Gradient Estimation
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            g = (1 - r1 - r2) * self.pop[idx].g + r1 * d_p + r2 * d_g
            g = g / (np.sum(g) + self.EPSILON)

            self.pop[idx].m = self.beta1 * self.pop[idx].m + (1 - self.beta1) * g
            self.pop[idx].v = self.beta2 * self.pop[idx].v + (1 - self.beta2) * g**2
            self.pop[idx].weights -= self.pop[idx].m / (np.sqrt(self.pop[idx].v) + self.EPSILON)

            # Advice Forward
            x_0 = self.pop[idx].solution + np.exp(-1.0 / (0.1 * self.epoch)) * 0.1 * hop * g
            x_0 = self.correct_solution(x_0)
            y_0 = self.get_target(x_0)

            # Random Search
            r3 = self.generator.uniform(-np.pi/2, np.pi/2, self.problem.n_dims)
            x_n = self.pop[idx].solution + np.tan(r3) * hop / epoch * 0.5
            x_n = self.correct_solution(x_n)
            y_n = self.get_target(x_n)

            # Encircling Mechanism
            d = self.pop[idx].local_solution - self.pop[idx].solution
            d_g = self.g_best.solution - self.pop[idx].solution
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            x_m = (1 - r1 - r2) * self.pop[idx].solution + r1 * d + r2 * d_g
            x_m = self.correct_solution(x_m)
            y_m = self.get_target(x_m)

            # Discriminant Condition
            y_list_compare = [y_0.fitness, y_n.fitness, y_m.fitness]
            y_list = [y_0, y_n, y_m]
            x_list = [x_0, x_n, x_m]
            if self.problem.minmax == "min":
                id_best = np.argmin(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]
            else:
                id_best = np.argmax(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]

            if self.compare_target(y_best, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].solution = x_best
                self.pop[idx].target = y_best
                if self.compare_target(y_best, self.pop[idx].local_target, self.problem.minmax):
                    self.pop[idx].local_solution = x_best
                    self.pop[idx].local_target = y_best
                    self.pop[idx].g = (np.sum(self.pop[idx].weights * self.pop[idx].solution) - self.pop[idx].target.fitness) * self.pop[idx].solution
            else:
                if self.generator.random() < 0.3:
                    self.pop[idx].solution = x_best
                    self.pop[idx].target = y_best
