#!/usr/bin/env python
# Created by "Thieu" at 07:44, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class ImprovedBSO(Optimizer):
    """
    The improved version: Improved Brain Storm Optimization (IBSO)

    Notes:
        + Remove some probability parameters, and some unnecessary equations.
        + The Levy-flight technique is employed to enhance the algorithm's robustness and resilience in challenging environments.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): 25% percent
        + p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
        + p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
        + p4 (float): [0.4, 0.6], Need more weights on the centers instead of the random position

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BSO
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
    >>> model = BSO.ImprovedBSO(epoch=1000, pop_size=50, m_clusters = 5, p1 = 0.25, p2 = 0.5, p3 = 0.75, p4 = 0.6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] El-Abd, M. (2017). Global-best brain storm optimization algorithm. Swarm and evolutionary computation, 37, 27-44.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, m_clusters: int = 5,
                 p1: float = 0.25, p2: float = 0.5, p3: float = 0.75, p4: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): 25% percent
            p2 (float): 50% percent changed by its own (local search), 50% percent changed by outside (global search)
            p3 (float): 75% percent develop the old idea, 25% invented new idea based on levy-flight
            p4 (float): Need more weights on the centers instead of the random position
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.m_clusters = self.validator.check_int("m_clusters", m_clusters, [2, int(self.pop_size/5)])
        self.p1 = self.validator.check_float("p1", p1, (0, 1.0))
        self.p2 = self.validator.check_float("p2", p2, (0, 1.0))
        self.p3 = self.validator.check_float("p3", p3, (0, 1.0))
        self.p4 = self.validator.check_float("p4", p4, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4"])
        self.sort_flag = False
        self.m_solution = int(self.pop_size / self.m_clusters)
        self.pop_group, self.centers = None, None

    def find_cluster__(self, pop_group):
        centers = []
        for idx in range(0, self.m_clusters):
            local_best = self.get_best_agent(pop_group[idx], self.problem.minmax)
            centers.append(local_best.copy())
        return centers

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.m_clusters, self.m_solution)
        self.centers = self.find_cluster__(self.pop_group)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        epsilon = 1. - 1. * epoch / self.epoch  # 1. Changed here, no need: k
        if self.generator.uniform() < self.p1:  # p_5a
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self.generate_agent()
        pop_group = self.pop_group
        for idx in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)

            if self.generator.uniform() < self.p2:  # p_6b
                if self.generator.uniform() < self.p3:
                    pos_new = self.centers[cluster_id].solution + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:  # 2. Using levy flight here
                    levy_step = self.get_levy_flight_step(beta=1.0, multiplier=0.001, size=self.problem.n_dims, case=-1)
                    pos_new = self.pop_group[cluster_id][location_id].solution + levy_step
            else:
                id1, id2 = self.generator.choice(range(0, self.m_clusters), 2, replace=False)
                if self.generator.uniform() < self.p4:
                    pos_new = 0.5 * (self.centers[id1].solution + self.centers[id2].solution) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_id1 = self.generator.integers(0, self.m_solution)
                    rand_id2 = self.generator.integers(0, self.m_solution)
                    pos_new = 0.5 * (self.pop_group[id1][rand_id1].solution + self.pop_group[id2][rand_id2].solution) + \
                              epsilon * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_group[cluster_id][location_id] = agent
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_group[cluster_id][location_id] = self.get_better_agent(agent, self.pop_group[cluster_id][location_id], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self.update_target_for_population(pop_group[idx])
                pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx], self.problem.minmax)

        # Needed to update the centers and population
        self.centers = self.find_cluster__(pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += pop_group[idx]


class OriginalBSO(ImprovedBSO):
    """
    The original version of: Brain Storm Optimization (BSO)

    Links:
        1. https://doi.org/10.1007/978-3-642-21515-5_36

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + m_clusters (int): [3, 10], number of clusters (m in the paper)
        + p1 (float): [0.1, 0.5], probability
        + p2 (float): [0.5, 0.95], probability
        + p3 (float): [0.2, 0.8], probability
        + p4 (float): [0.2, 0.8], probability
        + slope (int): [10, 15, 20, 25], changing logsig() function's slope (k: in the paper)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BSO
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
    >>> model = BSO.OriginalBSO(epoch=1000, pop_size=50, m_clusters = 5, p1 = 0.2, p2 = 0.8, p3 = 0.4, p4 = 0.5, slope = 20)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Y., 2011, June. Brain storm optimization algorithm. In International
    conference in swarm intelligence (pp. 303-309). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, m_clusters: int = 5, p1: float = 0.2,
                 p2: float = 0.8, p3: float = 0.4, p4: float = 0.5, slope: int = 20, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            m_clusters (int): number of clusters (m in the paper)
            p1 (float): probability
            p2 (float): probability
            p3 (float): probability
            p4 (float): probability
            slope (int): changing logsig() function's slope (k: in the paper)
        """
        super().__init__(epoch, pop_size, m_clusters, p1, p2, p3, p4, **kwargs)
        self.slope = self.validator.check_int("slope", slope, [10, 50])
        self.set_parameters(["epoch", "pop_size", "m_clusters", "p1", "p2", "p3", "p4", "slope"])

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rp = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rp)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        x = (0.5 * self.epoch - epoch) / self.slope
        epsilon = self.generator.uniform() * (1 / (1 + np.exp(-x)))
        if self.generator.random() < self.p1:  # p_5a
            idx = self.generator.integers(0, self.m_clusters)
            self.centers[idx] = self.generate_agent()
        pop_group = self.pop_group
        for idx in range(0, self.pop_size):  # Generate new individuals
            cluster_id = int(idx / self.m_solution)
            location_id = int(idx % self.m_solution)
            if self.generator.uniform() < self.p2:  # p_6b
                if self.generator.uniform() < self.p3:  # p_6i
                    cluster_id = self.generator.integers(0, self.m_clusters)
                if self.generator.uniform() < self.p3:
                    pos_new = self.centers[cluster_id].solution + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_idx = self.generator.integers(0, self.m_solution)
                    pos_new = self.pop_group[cluster_id][rand_idx].solution + self.generator.normal(0, 1, self.problem.n_dims)
            else:
                id1, id2 = self.generator.choice(range(0, self.m_clusters), 2, replace=False)
                if self.generator.uniform() < self.p4:
                    pos_new = 0.5 * (self.centers[id1].solution + self.centers[id2].solution) + epsilon * self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    rand_id1 = self.generator.integers(0, self.m_solution)
                    rand_id2 = self.generator.integers(0, self.m_solution)
                    pos_new = 0.5 * (self.pop_group[id1][rand_id1].solution + self.pop_group[id2][rand_id2].solution) + \
                              epsilon * self.generator.normal(0, 1, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_group[cluster_id][location_id] = agent
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop_group[cluster_id][location_id] = self.get_better_agent(agent, self.pop_group[cluster_id][location_id], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            for idx in range(0, self.m_clusters):
                pop_group[idx] = self.update_target_for_population(pop_group[idx])
                pop_group[idx] = self.greedy_selection_population(self.pop_group[idx], pop_group[idx], self.problem.minmax)
        # Needed to update the centers and population
        self.centers = self.find_cluster__(pop_group)
        self.pop = []
        for idx in range(0, self.m_clusters):
            self.pop += pop_group[idx]
