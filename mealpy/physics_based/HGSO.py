#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalHGSO(Optimizer):
    """
    The original version of: Henry Gas Solubility Optimization (HGSO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_clusters (int): [2, 10], number of clusters, default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, HGSO
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
    >>> model = HGSO.OriginalHGSO(epoch=1000, pop_size=50, n_clusters = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. Henry gas solubility
    optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, pp.646-667.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, n_clusters: int = 2, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_clusters (int): number of clusters, default = 2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.n_clusters = self.validator.check_int("n_clusters", n_clusters, [2, int(self.pop_size/5)])
        self.set_parameters(["epoch", "pop_size", "n_clusters"])
        self.n_elements = int(self.pop_size / self.n_clusters)
        self.sort_flag = False
        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epsilon = 0.05
        self.l1 = 5E-2
        self.l2 = 100.0
        self.l3 = 1E-2

    def initialize_variables(self):
        self.H_j = self.l1 * self.generator.uniform()
        self.P_ij = self.l2 * self.generator.uniform()
        self.C_j = self.l3 * self.generator.uniform()
        self.pop_group, self.p_best = None, None

    def initialization(self):
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_group = self.generate_group_population(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)  # multiple element

    def flatten_group__(self, group):
        pop = []
        for idx in range(0, self.n_clusters):
            pop += group[idx]
        return pop

    def get_best_solution_in_team__(self, group=None):
        list_best = []
        for idx in range(len(group)):
            best_agent = self.get_best_agent(group[idx], self.problem.minmax)
            list_best.append(best_agent)
        return list_best

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Loop based on the number of cluster in swarm (number of gases type)
        for idx in range(self.n_clusters):
            ### Loop based on the number of individual in each gases type
            pop_new = []
            for jdx in range(self.n_elements):
                F = -1.0 if self.generator.uniform() < 0.5 else 1.0
                ##### Based on Eq. 8, 9, 10
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(- ((self.p_best[idx].target.fitness + self.epsilon) / (self.pop_group[idx][jdx].target.fitness + self.epsilon)))
                pos_new = self.pop_group[idx][jdx].solution + F * self.generator.uniform() * gama * (self.p_best[idx].solution - self.pop_group[idx][jdx].solution) + \
                       F * self.generator.uniform() * self.alpha * (S_ij * self.g_best.solution - self.pop_group[idx][jdx].solution)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_new[-1].target = self.get_target(pos_new)
            pop_new = self.update_target_for_population(pop_new)
            self.pop_group[idx] = pop_new
        self.pop = self.flatten_group__(self.pop_group)

        ## Update Henry's coefficient using Eq.8
        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
        ## Update the solubility of each gas using Eq.9
        S_ij = self.K * self.H_j * self.P_ij
        ## Rank and select the number of worst agents using Eq. 11
        N_w = int(self.pop_size * (self.generator.uniform(0, 0.1) + 0.1))
        ## Update the position of the worst agents using Eq. 12
        sorted_id_pos = np.argsort([x.target.fitness for x in self.pop])

        pop_new = []
        pop_idx = []
        for item in range(N_w):
            id = sorted_id_pos[item]
            pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_idx.append(id)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx, id_selected in enumerate(pop_idx):
            self.pop[id_selected] = pop_new[idx].copy()
        self.pop_group = self.generate_group_population(self.pop, self.n_clusters, self.n_elements)
        self.p_best = self.get_best_solution_in_team__(self.pop_group)
