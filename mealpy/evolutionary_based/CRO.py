#!/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalCRO(Optimizer):
    """
    The original version of: Coral Reefs Optimization (CRO)

    Links:
        1. https://downloads.hindawi.com/journals/tswj/2014/739768.pdf

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.1, 0.7], Probability of depredation
        + GCR (float): [0.05, 0.2], probability for mutation process
        + gamma_min (float): [0.01, 0.1] factor for mutation process
        + gamma_max (float): [0.1, 0.5] factor for mutation process
        + n_trials (int): [2, 10], number of attempts for a larvar to set in the reef.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.CRO import OriginalCRO
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
    >>> po = 0.4
    >>> Fb = 0.9
    >>> Fa = 0.1
    >>> Fd = 0.1
    >>> Pd = 0.5
    >>> GCR = 0.1
    >>> gamma_min = 0.02
    >>> gamma_max = 0.2
    >>> n_trials = 5
    >>> model = OriginalCRO(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Salcedo-Sanz, S., Del Ser, J., Landa-Torres, I., Gil-López, S. and Portilla-Figueras, J.A., 2014.
    The coral reefs optimization algorithm: a novel metaheuristic for efficiently solving optimization problems. The Scientific World Journal, 2014.
    """

    def __init__(self, epoch=10000, pop_size=100,po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5, GCR=0.1,
                 gamma_min=0.02, gamma_max=0.2, n_trials=3, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            po (float): the rate between free/occupied at the beginning
            Fb (float): BroadcastSpawner/ExistingCorals rate
            Fa (float): fraction of corals duplicates its self and tries to settle in a different part of the reef
            Fd (float): fraction of the worse health corals in reef will be applied depredation
            Pd (float): the maximum of probability of depredation
            GCR (float): probability for mutation process
            gamma_min (float): factor for mutation process
            gamma_max (float): factor for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])  # ~ number of space
        self.po = self.validator.check_float("po", po, (0, 1.0))
        self.Fb = self.validator.check_float("Fb", Fb, (0, 1.0))
        self.Fa = self.validator.check_float("Fa", Fa, (0, 1.0))
        self.Fd = self.validator.check_float("Fd", Fd, (0, 1.0))
        self.Pd = self.validator.check_float("Pd", Pd, (0, 1.0))
        self.GCR = self.validator.check_float("GCR", GCR, (0, 1.0))
        self.gamma_min = self.validator.check_float("gamma_min", gamma_min, (0, 0.15))
        self.gamma_max = self.validator.check_float("gamma_max", gamma_max, (0.15, 1.0))
        self.n_trials = self.validator.check_int("n_trials", n_trials, [2, int(self.pop_size / 2)])
        self.set_parameters(["epoch", "pop_size", "po", "Fb", "Fa", "Fd", "Pd", "GCR", "gamma_min", "gamma_max", "n_trials"])
        self.sort_flag = False

    def initialization(self):
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        self.reef = np.array([])
        self.occupied_position = []  # after a gen, you should update the occupied_position
        self.G1 = self.gamma_max
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.gamma_max - self.gamma_min) / self.epoch
        self.num_occupied = int(self.pop_size / (1 + self.po))
        self.dyn_Pd = 0
        self.occupied_list = np.zeros(self.pop_size)
        self.occupied_idx_list = np.random.choice(list(range(self.pop_size)), self.num_occupied, replace=False)
        self.occupied_list[self.occupied_idx_list] = 1

    def gaussian_mutation__(self, position):
        random_pos = position + self.G1 * (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
        condition = np.random.random(self.problem.n_dims) < self.GCR
        pos_new = np.where(condition, random_pos, position)
        return self.amend_position(pos_new, self.problem.lb, self.problem.ub)

    ### Crossover
    def multi_point_cross__(self, pos1, pos2):
        p1, p2 = np.random.choice(list(range(len(pos1))), 2, replace=False)
        start, end = min(p1, p2), max(p1, p2)
        return np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)

    def larvae_setting__(self, larvae):
        # Trial to land on a square of reefs
        for larva in larvae:
            for i in range(self.n_trials):
                p = np.random.randint(0, self.pop_size - 1)
                if self.occupied_list[p] == 0:
                    self.pop[p] = larva
                    self.occupied_idx_list = np.append(self.occupied_idx_list, p)  # Update occupied id
                    self.occupied_list[p] = 1  # Update occupied list
                    break
                else:
                    if self.compare_agent(larva, self.pop[p]):
                        self.pop[p] = larva
                        break

    def sort_occupied_reef__(self):
        def reef_fitness(idx):
            return self.pop[idx][self.ID_TAR][self.ID_FIT]
        idx_list_sorted = sorted(self.occupied_idx_list, key=reef_fitness)
        return idx_list_sorted

    def broadcast_spawning_brooding__(self):
        # Step 1a
        larvae = []
        selected_corals = np.random.choice(self.occupied_idx_list, int(len(self.occupied_idx_list) * self.Fb), replace=False)
        for i in self.occupied_idx_list:
            if i not in selected_corals:
                pos_new = self.gaussian_mutation__(self.pop[i][self.ID_POS])
                larvae.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    larvae[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        # Step 1b
        while len(selected_corals) >= 2:
            id1, id2 = np.random.choice(range(len(selected_corals)), 2, replace=False)
            pos_new = self.multi_point_cross__(self.pop[selected_corals[id1]][self.ID_POS], self.pop[selected_corals[id2]][self.ID_POS])
            larvae.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                larvae[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            selected_corals = np.delete(selected_corals, [id1, id2])
        return self.update_target_wrapper_population(larvae)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)

        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_strim_population(pop_best, num_duplicate)
        self.larvae_setting__(pop_best)

        ## Depredation
        if np.random.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            self.occupied_idx_list = np.setdiff1d(self.occupied_idx_list, selected_depredator)
            for idx in selected_depredator:
                self.occupied_list[idx] = 0
        if self.dyn_Pd <= self.Pd:
            self.dyn_Pd += self.alpha
        if self.G1 >= self.gamma_min:
            self.G1 -= self.gama


class OCRO(OriginalCRO):
    """
    The original version of: Opposition-based Coral Reefs Optimization (OCRO)

    Links:
        1. https://dx.doi.org/10.2991/ijcis.d.190930.003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.1, 0.7], the maximum of probability of depredation
        + GCR (float): [0.05, 0.2], probability for mutation process
        + gamma_min (float): [0.01, 0.1] factor for mutation process
        + gamma_max (float): [0.1, 0.5] factor for mutation process
        + n_trials (int): [2, 10], number of attempts for a larvar to set in the reef
        + restart_count (int): [10, 100], reset the whole population after global best solution is not improved after restart_count times

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.CRO import OCRO
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
    >>> po = 0.4
    >>> Fb = 0.9
    >>> Fa = 0.1
    >>> Fd = 0.1
    >>> Pd = 0.5
    >>> GCR = 0.1
    >>> gamma_min = 0.02
    >>> gamma_max = 0.2
    >>> n_trials = 5
    >>> restart_count = 50
    >>> model = OCRO(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials, restart_count)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019. Efficient time-series forecasting using
    neural network and opposition-based coral reefs optimization. International Journal of Computational
    Intelligence Systems, 12(2), p.1144.
    """

    def __init__(self, epoch=10000, pop_size=100, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.5,
                 GCR=0.1, gamma_min=0.02, gamma_max=0.2, n_trials=3, restart_count=20, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            po (float): the rate between free/occupied at the beginning
            Fb (float): BroadcastSpawner/ExistingCorals rate
            Fa (float): fraction of corals duplicates its self and tries to settle in a different part of the reef
            Fd (float): fraction of the worse health corals in reef will be applied depredation
            Pd (float): Probability of depredation
            GCR (float): probability for mutation process
            gamma_min (float): [0.01, 0.1] factor for mutation process
            gamma_max (float): [0.1, 0.5] factor for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
            restart_count (int): reset the whole population after global best solution is not improved after restart_count times
        """
        super().__init__(epoch, pop_size, po, Fb, Fa, Fd, Pd, GCR, gamma_min, gamma_max, n_trials, **kwargs)
        self.restart_count = self.validator.check_int("restart_count", restart_count, [2, int(epoch / 2)])
        self.set_parameters(["epoch", "pop_size", "po", "Fb", "Fa", "Fd", "Pd", "GCR", "gamma_min", "gamma_max", "n_trials", "restart_count"])
        self.sort_flag = False

    def initialize_variables(self):
        self.reset_count = 0

    def local_search__(self, pop=None):
        pop_new = []
        for idx in range(0, len(pop)):
            random_pos = np.random.uniform(self.problem.lb, self.problem.ub)
            condition = np.random.random(self.problem.n_dims) < 0.5
            pos_new = np.where(condition, self.g_best[self.ID_POS], random_pos)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        return self.update_target_wrapper_population(pop_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding__()
        self.larvae_setting__(larvae)

        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_strim_population(pop_best, num_duplicate)
        pop_local_search = self.local_search__(pop_best)
        self.larvae_setting__(pop_local_search)

        ## Depredation
        if np.random.random() < self.dyn_Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self.sort_occupied_reef__()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            for idx in selected_depredator:
                ### Using opposition-based leanring
                oppo_pos = self.create_opposition_position(self.pop[idx], self.g_best)
                oppo_pos = self.amend_position(oppo_pos, self.problem.lb, self.problem.ub)
                oppo_reef = [oppo_pos, self.get_target_wrapper(oppo_pos)]
                if self.compare_agent(oppo_reef, self.pop[idx]):
                    self.pop[idx] = oppo_reef
                else:
                    self.occupied_idx_list = self.occupied_idx_list[~np.isin(self.occupied_idx_list, [idx])]
                    self.occupied_list[idx] = 0

        if self.dyn_Pd <= self.Pd:
            self.dyn_Pd += self.alpha
        if self.G1 >= self.gamma_min:
            self.G1 -= self.gama

        self.reset_count += 1
        _, local_best = self.get_global_best_solution(self.pop)
        if self.compare_agent(local_best, self.g_best):
            self.reset_count = 0

        if self.reset_count == self.restart_count:
            self.pop = self.create_population(self.pop_size)
            self.occupied_list = np.zeros(self.pop_size)
            self.occupied_idx_list = np.random.choice(range(self.pop_size), self.num_occupied, replace=False)
            self.occupied_list[self.occupied_idx_list] = 1
            self.reset_count = 0
