# !/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalBFO(Optimizer):
    """
    The original version of: Bacterial Foraging Optimization (BFO)

    Links:
        1. http://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html

    Notes
    ~~~~~
    + Ned and Nre parameters are replaced by epoch (generation)
    + The Nc parameter will also decreased to reduce the computation time.
    + Cost in this version equal to Fitness value in the paper.

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + Ci (float): [0.01, 0.3], step size, default=0.01
        + Ped (float): [0.1, 0.5], probability of elimination, default=0.25
        + Ned (int): elim_disp_steps (Removed), Ned=5,
        + Nre (int): reproduction_steps (Removed), Nre=50,
        + Nc (int): [3, 10], chem_steps (Reduce), Nc = Original Nc/2, default = 5
        + Ns (int): [2, 10], swim length, default=4
        + attract_repels (list): coefficient to calculate attract and repel force, default = (0.1, 0.2, 0.1, 10)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BFO import OriginalBFO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> Ci = 0.01
    >>> Ped = 0.25
    >>> Nc = 5
    >>> Ns = 4
    >>> attract_repels = [0.1, 0.2, 0.1, 10]
    >>> model = OriginalBFO(problem_dict1, epoch, pop_size, Ci, Ped, Nc, Ns, attract_repels)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Passino, K.M., 2002. Biomimicry of bacterial foraging for distributed optimization and control.
    IEEE control systems magazine, 22(3), pp.52-67.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_COST = 2
    ID_INTER = 3
    ID_SUM_NUTRIENTS = 4

    def __init__(self, problem, epoch=10000, pop_size=100,
                 Ci=0.01, Ped=0.25, Nc=5, Ns=4, attract_repels=(0.1, 0.2, 0.1, 10), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ci (float): step size, default=0.01
            Ped (float): p_eliminate, default=0.25
            Ned (int): elim_disp_steps (Removed)         Ned=5,
            Nre (int): reproduction_steps (Removed)      Nre=50,
            Nc (int): chem_steps (Reduce)                Nc = Original Nc/2, default = 5
            Ns (int): swim_length, default=4
            attract_repels (list): coefficient to calculate attract and repel force, default = (0.1, 0.2, 0.1, 10)
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.step_size = Ci
        self.p_eliminate = Ped
        self.chem_steps = Nc
        self.swim_length = Ns
        self.d_attr = attract_repels[0]
        self.w_attr = attract_repels[1]
        self.h_rep = attract_repels[2]
        self.w_rep = attract_repels[3]
        self.half_pop_size = int(self.pop_size / 2)

    def create_solution(self):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], cost, interaction, sum_nutrients]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        position = self.amend_position(position)
        fitness = self.get_fitness_position(position)
        cost = 0.0
        interaction = 0.0
        sum_nutrients = 0.0
        return [position, fitness, cost, interaction, sum_nutrients]

    def _compute_cell_interaction(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem.n_dims * ((cell[self.ID_POS] - other[self.ID_POS]) ** 2).mean(axis=None)
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def _attract_repel(self, idx, cells):
        attract = self._compute_cell_interaction(cells[idx], cells, -self.d_attr, -self.w_attr)
        repel = self._compute_cell_interaction(cells[idx], cells, self.h_rep, -self.w_rep)
        return attract + repel

    def _evaluate(self, idx, cells):
        cells[idx][self.ID_INTER] = self._attract_repel(idx, cells)
        cells[idx][self.ID_COST] = cells[idx][self.ID_TAR][self.ID_FIT] + cells[idx][self.ID_INTER]
        return cells

    def _tumble_cell(self, cell, step_size):
        delta_i = np.random.uniform(self.problem.lb, self.problem.ub)
        unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
        vector = cell[self.ID_POS] + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        for j in range(0, self.chem_steps):
            for idx in range(0, self.pop_size):
                sum_nutrients = 0.0
                self.pop = self._evaluate(idx, self.pop)
                sum_nutrients += self.pop[idx][self.ID_COST]

                for m in range(0, self.swim_length):
                    delta_i = np.random.uniform(self.problem.lb, self.problem.ub)
                    unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                    pos_new = self.pop[idx][self.ID_POS] + self.step_size * unit_vector
                    pos_new = self.amend_position(pos_new)
                    fit_new = self.get_fitness_position(pos_new)
                    nfe_epoch += 1
                    if self.compare_agent([pos_new, fit_new], self.pop[idx]):
                        self.pop[idx][self.ID_POS] = pos_new
                        self.pop[idx][self.ID_TAR] = fit_new
                        break
                    sum_nutrients += self.pop[idx][self.ID_COST]
                self.pop[idx][self.ID_SUM_NUTRIENTS] = sum_nutrients

            cells = sorted(self.pop, key=lambda cell: cell[self.ID_SUM_NUTRIENTS])
            self.pop = deepcopy(cells[0:self.half_pop_size]) + deepcopy(cells[0:self.half_pop_size])

            for idc in range(self.pop_size):
                if np.random.rand() < self.p_eliminate:
                    self.pop[idc] = self.create_solution()
                    nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch


class ABFO(Optimizer):
    """
    The original version of: Adaptive Bacterial Foraging Optimization (ABFO)

    Notes
    ~~~~~
    + This is the best improvement version of BFO
    + The population will remain the same length as initialization due to add and remove operators

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + Ci (list): C_s (start), C_e (end)  -=> step size # step size in BFO, default=(0.1, 0.001)
        + Ped (float): Probability eliminate, default=0.01
        + Ns (int): swim_length, default=4
        + N_minmax (list): (Dead threshold value, split threshold value), default=(2, 40)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BFO import ABFO
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> Ci = [0.1, 0.001]
    >>> Ped = 0.01
    >>> Ns = 4
    >>> N_minmax = [2, 40]
    >>> model = ABFO(problem_dict1, epoch, pop_size, Ci, Ped, Ns, N_minmax)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019, April. Building resource auto-scaler with functional-link
    neural network and adaptive bacterial foraging optimization. In International Conference on
    Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
    """

    ID_NUT = 2
    ID_LOC_POS = 3
    ID_LOC_FIT = 4

    def __init__(self, problem, epoch=10000, pop_size=100, Ci=(0.1, 0.001), Ped=0.01, Ns=4, N_minmax=(1, 40), **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ci (list): C_s (start), C_e (end)  -=> step size # step size in BFO, default=(0.1, 0.001)
            Ped (float): Probability eliminate, default=0.01
            Ns (int): swim_length, default=4
            N_minmax (list): (Dead threshold value, split threshold value), default=(2, 40)
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size
        self.step_size = Ci
        self.p_eliminate = Ped
        self.swim_length = Ns

        # (Dead threshold value, split threshold value) -> N_adapt, N_split
        self.N_adapt = N_minmax[0]  # Dead threshold value
        self.N_split = N_minmax[1]  # split threshold value

        self.C_s = self.step_size[0] * (self.problem.ub - self.problem.lb)
        self.C_e = self.step_size[1] * (self.problem.ub - self.problem.lb)

    def create_solution(self):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: wrapper of solution with format [position, [target, [obj1, obj2, ...]], nutrient, local_pos_best, local_fit_best]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        position = self.amend_position(position)
        fitness = self.get_fitness_position(position)
        nutrient = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_pos_best = deepcopy(position)
        local_fit_best = deepcopy(fitness)
        return [position, fitness, nutrient, local_pos_best, local_fit_best]

    def _update_step_size(self, pop=None, idx=None):
        total_fitness = np.sum(temp[self.ID_TAR][self.ID_FIT] for temp in pop)
        step_size = self.C_s - (self.C_s - self.C_e) * pop[idx][self.ID_TAR][self.ID_FIT] / total_fitness
        step_size = step_size / self.pop[idx][self.ID_NUT] if self.pop[idx][self.ID_NUT] > 0 else step_size
        return step_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        for i in range(0, self.pop_size):
            step_size = self._update_step_size(self.pop, i)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS]) + \
                          (self.pop[i][self.ID_LOC_POS] - self.pop[i][self.ID_POS])
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = np.random.uniform(self.problem.lb, self.problem.ub) if delta == 0 else (delta_i / delta)
                pos_new = self.pop[i][self.ID_POS] + step_size * unit_vector
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                nfe_epoch += 1
                if self.compare_agent([pos_new, fit_new], self.pop[i]):
                    self.pop[i][self.ID_POS] = pos_new
                    self.pop[i][self.ID_TAR] = fit_new
                    self.pop[i][self.ID_NUT] += 1
                    # Update personal best
                    if self.compare_agent([pos_new, fit_new], [None, self.pop[i][self.ID_LOC_FIT]]):
                        self.pop[i][self.ID_LOC_POS] = deepcopy(pos_new)
                        self.pop[i][self.ID_LOC_FIT] = deepcopy(fit_new)
                else:
                    self.pop[i][self.ID_NUT] -= 1

            if self.pop[i][self.ID_NUT] > max(self.N_split, self.N_split + (len(self.pop) - self.pop_size) / self.N_adapt):
                pos_new = self.pop[i][self.ID_POS] + np.random.normal(self.problem.lb, self.problem.ub) * \
                          (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                self.pop.append([pos_new, fit_new, 0, deepcopy(pos_new), deepcopy(fit_new)])
                nfe_epoch += 1

            nut_min = min(self.N_adapt, self.N_adapt + (len(self.pop) - self.pop_size) / self.N_adapt)
            if self.pop[i][self.ID_NUT] < nut_min or np.random.rand() < self.p_eliminate:
                self.pop[i] = self.create_solution()
                nfe_epoch += 1

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(self.pop):
            if tuple(obj[self.ID_POS].tolist()) in new_set:
                self.pop.pop(idx)
            else:
                new_set.add(tuple(obj[self.ID_POS].tolist()))

        ## Balance the population by adding more agents or remove some agents
        n_agents = len(self.pop) - self.pop_size
        if n_agents < 0:
            for idx in range(0, n_agents):
                self.pop.append(self.create_solution())
                nfe_epoch += 1
        elif n_agents > 0:
            list_idx_removed = np.random.choice(range(0, len(self.pop)), n_agents, replace=False)
            pop_new = []
            for idx in range(0, len(self.pop)):
                if idx not in list_idx_removed:
                    pop_new.append(self.pop[idx])
            self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch
