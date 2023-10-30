#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalBSA(Optimizer):
    """
    The original version of: Bird Swarm Algorithm (BSA)

    Links:
        1. https://doi.org/10.1080/0952813X.2015.1042530
        2. https://www.mathworks.com/matlabcentral/fileexchange/51256-bird-swarm-algorithm-bsa

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ff (int): (5, 20), flight frequency - default = 10
        + pff (float): the probability of foraging for food - default = 0.8
        + c_couples (list, tuple): [c1, c2] -> (2.0, 2.0), Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
        + a_couples (list, tuple): [a1, a2] -> (1.5, 1.5), The indirect and direct effect on the birds' vigilance behaviours.
        + fc (float): (0.1, 1.0), The followed coefficient - default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BSA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BSA.OriginalBSA(epoch=1000, pop_size=50, ff = 10, pff = 0.8, c1 = 1.5, c2 = 1.5, a1 = 1.0, a2 = 1.0, fc = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Meng, X.B., Gao, X.Z., Lu, L., Liu, Y. and Zhang, H., 2016. A new bio-inspired optimisation
    algorithm: Bird Swarm Algorithm. Journal of Experimental & Theoretical Artificial Intelligence, 28(4), pp.673-687.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, ff: int = 10, pff: float = 0.8,
                 c1: float = 1.5, c2: float = 1.5, a1: float = 1.0, a2: float = 1.0, fc: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ff (int): flight frequency - default = 10
            pff (float): the probability of foraging for food - default = 0.8
            c1 (float): Cognitive accelerated coefficient same as PSO
            c2 (float): Social accelerated coefficient same as PSO
            a1 (float): The indirect effect on the birds' vigilance behaviours.
            a2 (float): The direct effect on the birds' vigilance behaviours.
            fc (float): The followed coefficient - default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.ff = self.validator.check_int("ff", ff, [2, int(self.pop_size/2)])
        self.pff = self.validator.check_float("pff", pff, (0, 1.0))
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.a1 = self.validator.check_float("a1", a1, (0, 5.0))
        self.a2 = self.validator.check_float("a2", a2, (0, 5.0))
        self.fc = self.validator.check_float("fc", fc, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "ff", "pff", "c1", "c2", "a1", "a2", "fc"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_position = solution.copy()
        return Agent(solution=solution, local_solution=local_position)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pos_list = np.array([agent.solution for agent in self.pop])
        fit_list = np.array([agent.local_target.fitness for agent in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)

        if epoch % self.ff != 0:
            pop_new = []
            for idx in range(0, self.pop_size):
                agent = self.pop[idx].copy()
                prob = self.generator.uniform() * 0.2 + self.pff  # The probability of foraging for food
                if self.generator.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = self.pop[idx].solution + self.c1 * \
                            self.generator.uniform() * (self.pop[idx].local_solution - self.pop[idx].solution) + \
                            self.c2 * self.generator.uniform() * (self.g_best.solution - self.pop[idx].solution)
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a1 * np.exp(-self.pop_size * self.pop[idx].local_target.fitness / (self.EPSILON + fit_sum))
                    k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
                    t1 = (fit_list[idx] - fit_list[k]) / (abs(fit_list[idx] - fit_list[k]) + self.EPSILON)
                    A2 = self.a2 * np.exp(t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON))
                    x_new = self.pop[idx].solution + A1 * self.generator.uniform(0, 1) * (pos_mean - self.pop[idx].solution) + \
                            A2 * self.generator.uniform(-1, 1) * (self.g_best.solution - self.pop[idx].solution)
                agent.solution = self.correct_solution(x_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(agent.solution)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        else:
            pop_new = self.pop.copy()
            # Divide the bird swarm into two parts: producers and scroungers.
            min_idx = np.argmin(fit_list)
            max_idx = np.argmax(fit_list)
            choose = 0
            if min_idx < 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 1
            if min_idx > 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 2
            if min_idx < 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 3
            if min_idx > 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 4

            if choose < 3:  # Producing (Equation 5)
                for idx in range(int(self.pop_size / 2 + 1), self.pop_size):
                    agent = self.pop[idx].copy()
                    x_new = self.pop[idx].solution + self.generator.uniform(self.problem.lb, self.problem.ub) * self.pop[idx].solution
                    agent.solution = self.correct_solution(x_new)
                    pop_new[idx] = agent
                if choose == 1:
                    x_new = self.pop[min_idx].solution + self.generator.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx].solution
                    agent = self.pop[min_idx].copy()
                    agent.solution = self.correct_solution(x_new)
                    pop_new[min_idx] = agent
                for i in range(0, int(self.pop_size / 2)):
                    if choose == 2 or min_idx != i:
                        agent = self.pop[i].copy()
                        FL = self.generator.uniform() * 0.4 + self.fc
                        idx = self.generator.integers(0.5 * self.pop_size + 1, self.pop_size)
                        x_new = self.pop[i].solution + (self.pop[idx].solution - self.pop[i].solution) * FL
                        agent.solution = self.correct_solution(x_new)
                        pop_new[i] = agent
            else:  # Scrounging (Equation 6)
                for i in range(0, int(0.5 * self.pop_size)):
                    agent = self.pop[i].copy()
                    x_new = self.pop[i].solution + self.generator.uniform(self.problem.lb, self.problem.ub) * self.pop[i].solution
                    agent.solution = self.correct_solution(x_new)
                    pop_new[i] = agent
                if choose == 4:
                    agent = self.pop[min_idx].copy()
                    x_new = self.pop[min_idx].solution + self.generator.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx].solution
                    agent.solution = self.correct_solution(x_new)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    if choose == 3 or min_idx != i:
                        agent = self.pop[i].copy()
                        FL = self.generator.uniform() * 0.4 + self.fc
                        idx = self.generator.integers(0, 0.5 * self.pop_size)
                        x_new = self.pop[i].solution + (self.pop[idx].solution - self.pop[i].solution) * FL
                        agent.solution = self.correct_solution(x_new)
                        pop_new[i] = agent
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
            else:
                for idx in range(0, self.pop_size):
                    pop_new[idx].target = self.get_target(pop_new[idx].solution)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
