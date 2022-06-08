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

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ci (float): [0.01, 0.3], step size, default=0.01
        + Ped (float): [0.1, 0.5], probability of elimination, default=0.25
        + Ned (int): elim_disp_steps (Removed), Ned=5,
        + Nre (int): reproduction_steps (Removed), Nre=50,
        + Nc (int): [3, 10], chem_steps (Reduce), Nc = Original Nc/2, default = 5
        + Ns (int): [2, 10], swim length, default=4
        + attract_repels (list, tuple): coefficient to calculate attract and repel force, default = (0.1, 0.2, 0.1, 10)

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
            attract_repels (list, tuple): coefficient to calculate attract and repel force, default = (0.1, 0.2, 0.1, 10)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.step_size = self.Ci = self.validator.check_int("Ci", Ci, (0, 5.0))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.chem_steps = self.Nc = self.validator.check_int("Nc", Nc, [2, 100])
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.attract_repels = self.validator.check_tuple_float("attract_repels (d_attract, w_attract, h_repels, w_repels)",
                    attract_repels, ((0, 1.0), (0, 1.0), (0, 1.0), (2, 20)))
        self.d_attr, self.w_attr, self.h_rep, self.w_rep = self.attract_repels
        self.half_pop_size = int(self.pop_size / 2)
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, cost, interaction, sum_nutrients]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        cost = 0.0
        interaction = 0.0
        sum_nutrients = 0.0
        return [position, target, cost, interaction, sum_nutrients]

    def compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem.n_dims * ((cell[self.ID_POS] - other[self.ID_POS]) ** 2).mean(axis=None)
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def attract_repel__(self, idx, cells):
        attract = self.compute_cell_interaction__(cells[idx], cells, -self.d_attr, -self.w_attr)
        repel = self.compute_cell_interaction__(cells[idx], cells, self.h_rep, -self.w_rep)
        return attract + repel

    def evaluate__(self, idx, cells):
        cells[idx][self.ID_INTER] = self.attract_repel__(idx, cells)
        cells[idx][self.ID_COST] = cells[idx][self.ID_TAR][self.ID_FIT] + cells[idx][self.ID_INTER]
        return cells

    def tumble_cell__(self, cell, step_size):
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
                self.pop = self.evaluate__(idx, self.pop)
                sum_nutrients += self.pop[idx][self.ID_COST]

                for m in range(0, self.swim_length):
                    delta_i = np.random.uniform(self.problem.lb, self.problem.ub)
                    unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                    pos_new = self.pop[idx][self.ID_POS] + self.step_size * unit_vector
                    pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                    target = self.get_target_wrapper(pos_new)
                    nfe_epoch += 1
                    if self.compare_agent([pos_new, target], self.pop[idx]):
                        self.pop[idx][self.ID_POS] = pos_new
                        self.pop[idx][self.ID_TAR] = target
                        break
                    sum_nutrients += self.pop[idx][self.ID_COST]
                self.pop[idx][self.ID_SUM_NUTRIENTS] = sum_nutrients

            cells = sorted(self.pop, key=lambda cell: cell[self.ID_SUM_NUTRIENTS])
            self.pop = deepcopy(cells[0:self.half_pop_size]) + deepcopy(cells[0:self.half_pop_size])

            for idc in range(self.pop_size):
                if np.random.rand() < self.p_eliminate:
                    self.pop[idc] = self.create_solution(self.problem.lb, self.problem.ub)
                    nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch


class ABFO(Optimizer):
    """
    The original version of: Adaptive Bacterial Foraging Optimization (ABFO)

    Notes
    ~~~~~
    + This is the best improvement version of BFO
    + The population will remain the same length as initialization due to add and remove operators

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ci (list): C_s (start), C_e (end)  -=> step size # step size in BFO, default=(0.1, 0.001)
        + Ped (float): Probability eliminate, default=0.01
        + Ns (int): swim_length, default=4
        + N_minmax (list, tuple): (Dead threshold value, split threshold value), default=(2, 40)

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
            Ci (list, tuple): C_s (start), C_e (end)  -=> step size # step size in BFO, default=(0.1, 0.001)
            Ped (float): Probability eliminate, default=0.01
            Ns (int): swim_length, default=4
            N_minmax (list, tuple): (Dead threshold value, split threshold value), default=(2, 40)
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.step_size = self.Ci = self.validator.check_tuple_float("Ci (start, end)", Ci, ((0, 2.0), (0, 1.0)))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.N_minmax = self.validator.check_tuple_int("Threshold value (dead, split)", N_minmax, ([0, 4], [5, 50]))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False
        # (Dead threshold value, split threshold value) -> N_adapt, N_split
        self.N_adapt = self.N_minmax[0]  # Dead threshold value
        self.N_split = self.N_minmax[1]  # split threshold value

        self.C_s = self.step_size[0] * (self.problem.ub - self.problem.lb)
        self.C_e = self.step_size[1] * (self.problem.ub - self.problem.lb)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, nutrient, local_pos_best, local_fit_best]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        nutrient = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_pos_best = deepcopy(position)
        local_fit_best = deepcopy(target)
        return [position, target, nutrient, local_pos_best, local_fit_best]

    def update_step_size__(self, pop=None, idx=None):
        total_fitness = np.sum([temp[self.ID_TAR][self.ID_FIT] for temp in pop])
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
            step_size = self.update_step_size__(self.pop, i)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS]) + \
                          (self.pop[i][self.ID_LOC_POS] - self.pop[i][self.ID_POS])
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = np.random.uniform(self.problem.lb, self.problem.ub) if delta == 0 else (delta_i / delta)
                pos_new = self.pop[i][self.ID_POS] + step_size * unit_vector
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                nfe_epoch += 1
                if self.compare_agent([pos_new, target], self.pop[i]):
                    self.pop[i][self.ID_POS] = pos_new
                    self.pop[i][self.ID_TAR] = target
                    self.pop[i][self.ID_NUT] += 1
                    # Update personal best
                    if self.compare_agent([pos_new, target], [None, self.pop[i][self.ID_LOC_FIT]]):
                        self.pop[i][self.ID_LOC_POS] = deepcopy(pos_new)
                        self.pop[i][self.ID_LOC_FIT] = deepcopy(target)
                else:
                    self.pop[i][self.ID_NUT] -= 1

            if self.pop[i][self.ID_NUT] > max(self.N_split, self.N_split + (len(self.pop) - self.pop_size) / self.N_adapt):
                pos_new = self.pop[i][self.ID_POS] + np.random.normal(self.problem.lb, self.problem.ub) * \
                          (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)
                self.pop.append([pos_new, target, 0, deepcopy(pos_new), deepcopy(target)])
                nfe_epoch += 1

            nut_min = min(self.N_adapt, self.N_adapt + (len(self.pop) - self.pop_size) / self.N_adapt)
            if self.pop[i][self.ID_NUT] < nut_min or np.random.rand() < self.p_eliminate:
                self.pop[i] = self.create_solution(self.problem.lb, self.problem.ub)
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
                self.pop.append(self.create_solution(self.problem.lb, self.problem.ub))
                nfe_epoch += 1
        elif n_agents > 0:
            list_idx_removed = np.random.choice(range(0, len(self.pop)), n_agents, replace=False)
            pop_new = []
            for idx in range(0, len(self.pop)):
                if idx not in list_idx_removed:
                    pop_new.append(self.pop[idx])
            self.pop = pop_new
        self.nfe_per_epoch = nfe_epoch
