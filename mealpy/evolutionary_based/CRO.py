# !/usr/bin/env python
# Created by "Thieu" at 14:56, 19/11/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseCRO(Optimizer):
    """
    The original version of: Coral Reefs Optimization (CRO)

    Links:
        1. http://downloads.hindawi.com/journals/tswj/2014/739768.pdf

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.05, 0.3], Probability of depredation
        + G (list): (gamma_min, gamma_max) -> ([0.01, 0.1], [0.1, 0.5]), factor for mutation process
        + GCR (float): [0.05, 0.2], probability for mutation process
        + n_trials (int): [2, 10], number of attempts for a larvar to set in the reef.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.CRO import BaseCRO
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
    >>> po = 0.4
    >>> Fb = 0.9
    >>> Fa = 0.1
    >>> Fd = 0.1
    >>> Pd = 0.1
    >>> G = [0.02, 0.2]
    >>> GCR = 0.1
    >>> n_trials = 5
    >>> model = BaseCRO(problem_dict1, epoch, pop_size, po, Fb, Fa, Fd, Pd, G, GCR, n_trials)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Salcedo-Sanz, S., Del Ser, J., Landa-Torres, I., Gil-López, S. and Portilla-Figueras, J.A., 2014.
    The coral reefs optimization algorithm: a novel metaheuristic for efficiently solving optimization problems. The Scientific World Journal, 2014.
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=(0.02, 0.2), GCR=0.1, n_trials=3, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            po (float): the rate between free/occupied at the beginning
            Fb (float): BroadcastSpawner/ExistingCorals rate
            Fa (float): fraction of corals duplicates its self and tries to settle in a different part of the reef
            Fd (float): fraction of the worse health corals in reef will be applied depredation
            Pd (float): Probability of depredation
            G (list): (gamma_min, gamma_max), factor for mutation process
            GCR (float): probability for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False

        self.epoch = epoch
        self.pop_size = pop_size  # ~ number of space
        self.po = po
        self.Fb = Fb
        self.Fa = Fa
        self.Fd = Fd
        self.Pd_threshold = Pd
        self.Pd = 0
        self.n_trials = n_trials
        self.G = G
        self.GCR = GCR
        self.G1 = G[1]
        self.reef = np.array([])
        self.occupied_position = []  # after a gen, you should update the occupied_position
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.G[1] - self.G[0]) / self.epoch
        self.num_occupied = int(self.pop_size / (1 + self.po))

        self.occupied_list = np.zeros(self.pop_size)
        self.occupied_idx_list = np.random.choice(list(range(self.pop_size)), self.num_occupied, replace=False)
        self.occupied_list[self.occupied_idx_list] = 1

    def _gaussian_mutation(self, position):
        temp = position + self.G1 * (self.problem.ub - self.problem.lb) * np.random.normal(0, 1, self.problem.n_dims)
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.GCR, temp, position)
        return self.amend_position(pos_new)

    ### Crossover
    def _multi_point_cross(self, pos1, pos2):
        p1, p2 = np.random.choice(list(range(len(pos1))), 2, replace=False)
        start = min(p1, p2)
        end = max(p1, p2)
        return np.concatenate((pos1[:start], pos2[start:end], pos1[end:]), axis=0)

    def _larvae_setting(self, larvae):
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

    def _sort_occupied_reef(self):
        def reef_fitness(idx):
            return self.pop[idx][self.ID_TAR][self.ID_FIT]

        idx_list_sorted = sorted(self.occupied_idx_list, key=reef_fitness)
        return idx_list_sorted

    def broadcast_spawning_brooding(self):
        # Step 1a
        larvae = []
        selected_corals = np.random.choice(self.occupied_idx_list, int(len(self.occupied_idx_list) * self.Fb), replace=False)
        for i in self.occupied_idx_list:
            if i not in selected_corals:
                pos_new = self._gaussian_mutation(self.pop[i][self.ID_POS])
                larvae.append([pos_new, None])
        # Step 1b
        while len(selected_corals) >= 2:
            id1, id2 = np.random.choice(range(len(selected_corals)), 2, replace=False)
            pos_new = self._multi_point_cross(self.pop[selected_corals[id1]][self.ID_POS], self.pop[selected_corals[id2]][self.ID_POS])
            larvae.append([pos_new, None])
            selected_corals = np.delete(selected_corals, [id1, id2])
        return self.update_fitness_population(larvae)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding()
        self._larvae_setting(larvae)
        nfe_epoch += len(larvae)

        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_strim_population(pop_best, num_duplicate)
        self._larvae_setting(pop_best)

        ## Depredation
        if np.random.random() < self.Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self._sort_occupied_reef()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            for idx in selected_depredator:
                del self.occupied_idx_list[idx]
                self.occupied_list[idx] = 0

        if self.Pd <= self.Pd_threshold:
            self.Pd += self.alpha
        if self.G1 >= self.G[0]:
            self.G1 -= self.gama
        self.nfe_per_epoch = nfe_epoch


class OCRO(BaseCRO):
    """
    The original version of: Opposition-based Coral Reefs Optimization (OCRO)

    Links:
        1. https://dx.doi.org/10.2991/ijcis.d.190930.003

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + po (float): [0.2, 0.5], the rate between free/occupied at the beginning
        + Fb (float): [0.6, 0.9], BroadcastSpawner/ExistingCorals rate
        + Fa (float): [0.05, 0.3], fraction of corals duplicates its self and tries to settle in a different part of the reef
        + Fd (float): [0.05, 0.5], fraction of the worse health corals in reef will be applied depredation
        + Pd (float): [0.05, 0.3], Probability of depredation
        + G (list): (gamma_min, gamma_max) -> ([0.01, 0.1], [0.1, 0.5]), factor for mutation process
        + GCR (float): [0.05, 0.2], probability for mutation process
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
    >>>     "verbose": True,
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> po = 0.4
    >>> Fb = 0.9
    >>> Fa = 0.1
    >>> Fd = 0.1
    >>> Pd = 0.1
    >>> G = [0.02, 0.2]
    >>> GCR = 0.1
    >>> n_trials = 5
    >>> restart_count = 50
    >>> model = OCRO(problem_dict1, epoch, pop_size, po, Fb, Fa, Fd, Pd, G, GCR, n_trials, restart_count)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019. Efficient time-series forecasting using
    neural network and opposition-based coral reefs optimization. International Journal of Computational
    Intelligence Systems, 12(2), p.1144.
    """

    def __init__(self, problem, epoch=10000, pop_size=100,
                 po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=(0.02, 0.2), GCR=0.1, n_trials=3, restart_count=55, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            po (float): the rate between free/occupied at the beginning
            Fb (float): BroadcastSpawner/ExistingCorals rate
            Fa (float): fraction of corals duplicates its self and tries to settle in a different part of the reef
            Fd (float): fraction of the worse health corals in reef will be applied depredation
            Pd (float): Probability of depredation
            G (list): (gamma_min, gamma_max), factor for mutation process
            GCR (float): probability for mutation process
            n_trials (int): number of attempts for a larva to set in the reef.
            restart_count (int): reset the whole population after global best solution is not improved after restart_count times
        """
        super().__init__(problem, epoch, pop_size, po, Fb, Fa, Fd, Pd, G, GCR, n_trials, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False
        self.restart_count = restart_count
        self.reset_count = 0

    def _local_search(self, pop=None):
        pop_new = []
        for idx in range(0, len(pop)):
            temp = np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < 0.5, self.g_best[self.ID_POS], temp)
            pos_new = self.amend_position(pos_new)
            pop_new.append([pos_new, None])
        return self.update_fitness_population(pop_new)

    def _opposition_based_position(self, reef, g_best):
        pos_new = self.problem.ub + self.problem.lb - g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - reef[self.ID_POS])
        pos_new = self.amend_position(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        return [pos_new, fit_new]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        ## Broadcast Spawning Brooding
        larvae = self.broadcast_spawning_brooding()
        self._larvae_setting(larvae)
        nfe_epoch += len(larvae)

        ## Asexual Reproduction
        num_duplicate = int(len(self.occupied_idx_list) * self.Fa)
        pop_best = [self.pop[idx] for idx in self.occupied_idx_list]
        pop_best = self.get_sorted_strim_population(pop_best, num_duplicate)
        pop_local_search = self._local_search(pop_best)
        self._larvae_setting(pop_local_search)

        ## Depredation
        if np.random.random() < self.Pd:
            num__depredation__ = int(len(self.occupied_idx_list) * self.Fd)
            idx_list_sorted = self._sort_occupied_reef()
            selected_depredator = idx_list_sorted[-num__depredation__:]
            for idx in selected_depredator:
                opposite_reef = self._opposition_based_position(self.pop[idx], self.g_best)
                if self.compare_agent(opposite_reef, self.pop[idx]):
                    self.pop[idx] = opposite_reef
                else:
                    del self.occupied_idx_list[idx]
                    self.occupied_list[idx] = 0

        if self.Pd <= self.Pd_threshold:
            self.Pd += self.alpha
        if self.G1 >= self.G[0]:
            self.G1 -= self.gama

        self.reset_count += 1
        _, local_best = self.get_global_best_solution(self.pop)
        if self.compare_agent(local_best, self.g_best):
            self.reset_count = 0

        if self.reset_count == self.restart_count:
            nfe_epoch += self.pop_size
            self.pop = self.create_population(self.pop_size)
            self.occupied_list = np.zeros(self.pop_size)
            self.occupied_idx_list = np.random.choice(range(self.pop_size), self.num_occupied, replace=False)
            self.occupied_list[self.occupied_idx_list] = 1
            self.reset_count = 0
        self.nfe_per_epoch = nfe_epoch
