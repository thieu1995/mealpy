#!/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalBFO(Optimizer):
    """
    The original version of: Bacterial Foraging Optimization (BFO)

    Notes:
        + Ned and Nre parameters are replaced by epoch (generation)
        + The Nc parameter will also decrease to reduce the computation time.
        + Cost in this version equal to Fitness value in the paper.
        + https://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ci (float): [0.01, 0.3], step size, default=0.01
        + Ped (float): [0.1, 0.5], probability of elimination, default=0.25
        + Ned (int): elim_disp_steps (Removed), Ned=5,
        + Nre (int): reproduction_steps (Removed), Nre=50,
        + Nc (int): [3, 10], chem_steps (Reduce), Nc = Original Nc/2, default = 5
        + Ns (int): [2, 10], swim length, default=4
        + d_attract (float): coefficient to calculate attract force, default = 0.1
        + w_attract (float): coefficient to calculate attract force, default = 0.2
        + h_repels (float): coefficient to calculate repel force, default = 0.1
        + w_repels (float): coefficient to calculate repel force, default = 10

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BFO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BFO.OriginalBFO(epoch=1000, pop_size=50, Ci = 0.01, Ped = 0.25, Nc = 5, Ns = 4, d_attract=0.1, w_attract=0.2, h_repels=0.1, w_repels=10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Passino, K.M., 2002. Biomimicry of bacterial foraging for distributed optimization and control.
    IEEE control systems magazine, 22(3), pp.52-67.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, Ci: float = 0.01, Ped: float = 0.25, Nc: int = 5, Ns: int = 4,
                 d_attract: float = 0.1, w_attract: float = 0.2, h_repels: float = 0.1, w_repels: float = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ci (float): step size, default=0.01
            Ped (float): p_eliminate, default=0.25
            Ned (int): elim_disp_steps (Removed)         Ned=5,
            Nre (int): reproduction_steps (Removed)      Nre=50,
            Nc (int): chem_steps (Reduce)                Nc = Original Nc/2, default = 5
            Ns (int): swim_length, default=4
            d_attract (float): coefficient to calculate attract force, default = 0.1
            w_attract (float): coefficient to calculate attract force, default = 0.2
            h_repels (float): coefficient to calculate repel force, default = 0.1
            w_repels (float): coefficient to calculate repel force, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.step_size = self.Ci = self.validator.check_int("Ci", Ci, (0, 5.0))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.chem_steps = self.Nc = self.validator.check_int("Nc", Nc, [2, 100])
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.d_attract = self.validator.check_float("d_attract", d_attract, (0, 1.0))
        self.w_attract = self.validator.check_float("w_attract", w_attract, (0, 1.0))
        self.h_repels = self.validator.check_float("h_repels", h_repels, (0, 1.0))
        self.w_repels = self.validator.check_float("w_repels", w_repels, (2.0, 20.0))
        self.set_parameters(["epoch", "pop_size", "Ci", "Ped", "Nc", "Ns", "d_attract", "w_attract", "h_repels", "w_repels"])
        self.half_pop_size = int(self.pop_size / 2)
        self.is_parallelizable = False
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        cost = 0.0
        interaction = 0.0
        nutrients = 0.0
        return Agent(solution=solution, cost=cost, interaction=interaction, nutrients=nutrients)

    def compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem.n_dims * ((cell.solution - other.solution) ** 2).mean(axis=None)
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def attract_repel__(self, idx, cells):
        attract = self.compute_cell_interaction__(cells[idx], cells, -self.d_attract, -self.w_attract)
        repel = self.compute_cell_interaction__(cells[idx], cells, self.h_repels, -self.w_repels)
        return attract + repel

    def evaluate__(self, idx, cells):
        cells[idx].interaction = self.attract_repel__(idx, cells)
        cells[idx].cost = cells[idx].target.fitness + cells[idx].interaction
        return cells

    def tumble_cell__(self, cell, step_size):
        delta_i = self.generator.uniform(self.problem.lb, self.problem.ub)
        unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
        vector = cell.solution + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for j in range(0, self.chem_steps):
            for idx in range(0, self.pop_size):
                sum_nutrients = 0.0
                self.pop = self.evaluate__(idx, self.pop)
                sum_nutrients += self.pop[idx].cost

                for m in range(0, self.swim_length):
                    delta_i = self.generator.uniform(self.problem.lb, self.problem.ub)
                    unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                    pos_new = self.pop[idx].solution + self.step_size * unit_vector
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                        self.pop[idx] = agent
                        break
                    sum_nutrients += self.pop[idx].cost
                self.pop[idx].nutrients = sum_nutrients
            cells = sorted(self.pop, key=lambda cell: cell.nutrients)
            self.pop = cells[0:self.half_pop_size].copy() + cells[0:self.half_pop_size].copy()
            for idc in range(self.pop_size):
                if self.generator.random() < self.p_eliminate:
                    self.pop[idc] = self.generate_agent()


class ABFO(Optimizer):
    """
    The original version of: Adaptive Bacterial Foraging Optimization (ABFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:

        + C_s (float): step size start, default=0.1
        + C_e (float): step size end, default=0.001
        + Ped (float): Probability eliminate, default=0.01
        + Ns (int): swim_length, default=4
        + N_adapt (int): Dead threshold value default=2
        + N_split (int): Split threshold value, default=40

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BFO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BFO.ABFO(epoch=1000, pop_size=50, C_s=0.1, C_e=0.001, Ped = 0.01, Ns = 4, N_adapt = 2, N_split = 40)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019, April. Building resource auto-scaler with functional-link
    neural network and adaptive bacterial foraging optimization. In International Conference on
    Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, C_s: float = 0.1, C_e: float = 0.001,
                 Ped: float = 0.01, Ns: int = 4, N_adapt: int = 2, N_split: int = 40, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            C_s (float): step size start, default=0.1
            C_e (float): step size end, default=0.001
            Ped (float): Probability eliminate, default=0.01
            Ns (int): swim_length, default=4
            N_adapt (int): Dead threshold value default=2
            N_split (int): Split threshold value, default=40
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.C_s = self.Ped = self.validator.check_float("C_s", C_s, (0, 2.0))
        self.C_e = self.Ped = self.validator.check_float("C_e", C_e, (0, 1.0))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.N_adapt = self.validator.check_int("N_adapt", N_adapt, [0, 4])
        self.N_split = self.validator.check_int("N_split", N_split, [5, 50])
        self.set_parameters(["epoch", "pop_size", "C_s", "C_e", "Ped", "Ns", "N_adapt", "N_split"])
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.C_s = self.C_s * (self.problem.ub - self.problem.lb)
        self.C_e = self.C_e * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        nutrients = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_solution = solution.copy()
        return Agent(solution=solution, nutrients=nutrients, local_solution=local_solution)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def update_step_size__(self, pop=None, idx=None):
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        step_size = self.C_s - (self.C_s - self.C_e) * pop[idx].target.fitness / total_fitness
        step_size = step_size / self.pop[idx].nutrients if self.pop[idx].nutrients > 0 else step_size
        return step_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            step_size = self.update_step_size__(self.pop, idx)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best.solution - self.pop[idx].solution) + (self.pop[idx].local_solution - self.pop[idx].solution)
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = self.generator.uniform(self.problem.lb, self.problem.ub) if delta == 0 else (delta_i / delta)
                pos_new = self.pop[idx].solution + step_size * unit_vector
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
                    agent.nutrients += 1
                    self.pop[idx] = agent
                    # Update personal best
                    if self.compare_target(agent.target, self.pop[idx].local_target, self.problem.minmax):
                        self.pop[idx].update(local_solution=pos_new.copy(), local_target=agent.target.copy())
                else:
                    self.pop[idx].nutrients -= 1
            if self.pop[idx].nutrients > max(self.N_split, self.N_split + (len(self.pop) - self.pop_size) / self.N_adapt):
                tt = self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = tt * self.pop[idx].solution + (1 - tt) * (self.g_best.solution - self.pop[idx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                self.pop.append(agent)
            nut_min = min(self.N_adapt, self.N_adapt + (len(self.pop) - self.pop_size) / self.N_adapt)
            if self.pop[idx].nutrients < nut_min or self.generator.random() < self.p_eliminate:
                self.pop[idx] = self.generate_agent()
        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(self.pop):
            if tuple(obj.solution.tolist()) in new_set:
                self.pop.pop(idx)
            else:
                new_set.add(tuple(obj.solution.tolist()))
        ## Balance the population by adding more agents or remove some agents
        n_agents = len(self.pop) - self.pop_size
        if n_agents < 0:
            for idx in range(0, n_agents):
                agent = self.generate_agent()
                self.pop.append(agent)
        elif n_agents > 0:
            list_idx_removed = self.generator.choice(range(0, len(self.pop)), n_agents, replace=False)
            pop_new = []
            for idx in range(0, len(self.pop)):
                if idx not in list_idx_removed:
                    pop_new.append(self.pop[idx])
            self.pop = pop_new
