#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 02:01, 07/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import array, take, where, concatenate, inf
from random import sample, random, randint
from numpy.random import uniform, normal
from mealpy.root import Root


class BaseCRO(Root):
    """
        The original version of: Coral Reefs Optimization (CRO)
            http://downloads.hindawi.com/journals/tswj/2014/739768.pdf
    """
    HEALTH = inf

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=(0.02, 0.2), GCR=0.1, k=3, **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        # reef_size: size of the reef, NxM square grids, each  grid stores a position
        # po: the rate between free/occupied at the beginning
        # Fb: BroadcastSpawner/ExistingCorals rate
        # Fa: fraction of corals duplicates its self and tries to settle in a different part of the reef
        # Fd: fraction of the worse health corals in reef will be applied depredation
        # Pd: Probabilty of depredation
        # k : number of attempts for a larvar to set in the reef.
        # reef: a maxtrix of dictionaries, each of those store a space's information (occupied/position/health)
        # occupied_corals: position of occupied corals in reef-matrix (array 1dimension, each element store a position)
        # unselected_corals: corals in occupied_corals that aren't selected in broadcastSpawning
        # larvae: all larva ready to setting
        # sorted_health: a list of position, refer to position of each position in reef-matrix, was sorted according to coral's health
        self.epoch = epoch
        self.pop_size = pop_size  # ~ number of space
        self.po = po
        self.Fb = Fb
        self.Fa = Fa
        self.Fd = Fd
        self.Pd_thres = Pd
        self.Pd = 0
        self.k = k
        self.G = G
        self.GCR = GCR
        self.G1 = G[1]
        self.reef = array([])
        self.occupied_position = []                 # after a gen, you should update the occupied_position
        self.alpha = 10 * self.Pd / self.epoch
        self.gama = 10 * (self.G[1] - self.G[0]) / self.epoch

    # Init the coral reefs
    def _init_reef__(self):
        reef = array([{'occupied': 0, 'position': array([]), 'health': self.HEALTH} for _ in range(self.pop_size)])
        num_occupied = int(self.pop_size / (1 + self.po))
        occupied_position = sample(list(range(self.pop_size)), num_occupied)
        for i in occupied_position:
            reef[i]['occupied'] = 1
            reef[i]['position'] = uniform(self.lb, self.ub)
            reef[i]['health'] = self.get_fitness_position(reef[i]['position'])
        self.occupied_position = occupied_position
        self.reef = reef

    # Update position which has been occupied by coral
    def _update_occupied_position__(self):
        self.occupied_position = []
        for i in range(self.pop_size):
            if self.reef[i]['occupied'] == 1:
                self.occupied_position.append(i)

    def _sort_occupied_position__(self):
        def referHealth(location):
            return self.reef[location]['health']
        self.occupied_position.sort(key=referHealth)

    def _broadcast_spawning_brooding__(self):
        # Step 1a
        self._update_occupied_position__()
        larvae = []
        num_of_occupied = len(self.occupied_position)
        selected_corals = sample(self.occupied_position, int(num_of_occupied * self.Fb))
        for i in self.occupied_position:
            if i not in selected_corals:
                p_solution = self.reef[i]['position']
                blarva = self._gausion_mutation__(p_solution)
                larvae.append(blarva)
        # Step 1b
        while len(selected_corals) >= 2:
            p1, p2 = sample(selected_corals, 2)
            p1_solution = self.reef[p1]['position']
            p2_solution = self.reef[p2]['position']
            larva = self._multi_point_cross__(p1_solution, p2_solution)
            larvae.append(larva)
            selected_corals.remove(p1)
            selected_corals.remove(p2)
        self._larvae_setting__(larvae)

    def _larvae_setting__(self, larvae):
        for larva in larvae:
            larva_fit = self.get_fitness_position(larva)
            for i in range(self.k):
                p = randint(0, self.pop_size - 1)
                if self.reef[p]['occupied'] == 0 or self.reef[p]['health'] > larva_fit:
                    self.reef[p]['occupied'] = 1
                    self.reef[p]['position'] = larva
                    self.reef[p]['health'] = larva_fit
                    break

    def _budding_setting__(self, larvae):
        for larva in larvae:
            for i in range(self.k):
                p = randint(0, self.pop_size - 1)
                if self.reef[p]['occupied'] == 0 or (self.reef[p]['health'] > larva['health']):
                    self.reef[p] = larva.copy()
                    break

    def _asexual_reproduction__(self):
        self._update_occupied_position__()              # Update the position
        self._sort_occupied_position__()
        num_duplicate = int(len(self.occupied_position) * self.Fa)
        selected_duplicator = self.occupied_position[:num_duplicate]
        duplicated_corals = take(self.reef, selected_duplicator)
        self._budding_setting__(duplicated_corals)

    def _depredation__(self):
        rate = random()
        if rate < self.Pd:
            num__depredation__ = int(len(self.occupied_position) * self.Fd)
            selected_depredator = self.occupied_position[-num__depredation__:]
            for pos in selected_depredator:
                self.reef[pos]['occupied'] = 0

    ### Crossover
    def _multi_point_cross__(self, parent1, parent2):
        p1, p2 = sample(list(range(len(parent1))), 2)
        start = min(p1, p2)
        end = max(p1, p2)
        return concatenate((parent1[:start], parent2[start:end], parent1[end:]), axis=0)

    def _gausion_mutation__(self, solution):
        temp = solution + self.G1 * (self.ub - self.lb) * normal(0, 1, self.problem_size)
        pos_new = where(uniform(0, 1, self.problem_size) < self.GCR, temp, solution)
        return self.amend_position_faster(pos_new)

    def train(self):
        g_best = {"occupied": 0, "position": None, "health": self.HEALTH}
        self._init_reef__()
        for epoch in range(0, self.epoch):
            self._broadcast_spawning_brooding__()
            self._asexual_reproduction__()
            self._depredation__()
            if self.Pd <= self.Pd_thres:
                self.Pd += self.alpha
            if self.G1 >= self.G[0]:
                self.G1 -= self.gama
            current_best = self.reef[self.occupied_position[0]]
            if current_best['health'] < g_best["health"]:
                g_best = current_best
            self.loss_train.append(g_best["health"])
            if self.verbose:
                print("> Epoch {}: Best fit: {}".format(epoch + 1, g_best["health"]))
        self.solution = [g_best["position"], g_best['health']]
        return g_best["position"], g_best['health'], self.loss_train


class OCRO(BaseCRO):
    """
        This is a variant of CRO which is combined Opposition-based learning and Coral Reefs Optimization
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, G=(0.02, 0.2), GCR=0.1, k=3, restart_count=55, **kwargs):
        BaseCRO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, po, Fb, Fa, Fd, Pd, G, GCR, k, kwargs=kwargs)
        self.best_sol = []
        self.restart_count = restart_count

    def __local_seach__(self):
        reef = array([{'occupied': 0, 'position': [], 'health': self.HEALTH} for i in range(self.pop_size)])
        num_occupied = int(self.pop_size / (1 + self.po))
        occupied_position = sample(list(range(self.pop_size)), num_occupied)
        for i in occupied_position:
            reef[i]['occupied'] = 1
            temp = uniform(self.lb, self.ub)
            reef[i]['position'] = where(uniform(0, 1, self.problem_size) < 0.1, self.best_sol, temp)
            reef[i]['health'] = self.get_fitness_position(reef[i]['position'])
        self.occupied_position = occupied_position
        self.reef = reef

    def __opposition_based_larva__(self, budding_pos):
        larvae = []
        for i in budding_pos:
            Olarva = self.reef[i].copy()
            Olarva['position'] = self.ub + self.lb - self.best_sol + random() * (self.best_sol - self.reef[i]['position'])
            Olarva['health'] = self.get_fitness_position(Olarva['position'])
            if Olarva['health'] < self.reef[i]['health']:
                larvae.append(Olarva)
        return larvae

    def _asexual_reproduction__(self):
        self._update_occupied_position__()
        self._sort_occupied_position__()
        self.best_coral = self.reef[self.occupied_position[0]]
        num_duplicate = int(len(self.occupied_position) * self.Fa)
        selected_duplicator = self.occupied_position[:num_duplicate]
        duplicated_corals = take(self.reef, selected_duplicator)
        self._budding_setting__(duplicated_corals)

    def _depredation__(self):
        num_depredation = int(len(self.occupied_position) * self.Fd)
        selected_depredator = sample(self.occupied_position[-self.pop_size // 2:], num_depredation)
        for i in selected_depredator:
            Olarva = self.reef[i].copy()
            Olarva['position'] = (self.ub + self.lb - self.best_sol + uniform(0, 1, self.problem_size) * (self.best_sol - self.reef[i]['position']))
            Olarva['health'] = self.get_fitness_position(Olarva['position'])
            if Olarva['health'] < self.reef[i]['health']:
                self.reef[i] = Olarva.copy()
            else:
                self.reef[i]['occupied'] = 0

    def train(self):
        g_best = {"occupied": 0, "position": array([]), "health": self.HEALTH}
        self._init_reef__()
        self.best_sol = array(self.reef[self.occupied_position[0]]['position']).copy()
        reset_count = 0
        for epoch in range(0, self.epoch):
            self._broadcast_spawning_brooding__()
            self._asexual_reproduction__()
            self._depredation__()
            if self.Pd <= self.Pd_thres:
                self.Pd += self.alpha
            if self.G1 >= self.G[0]:
                self.G1 -= self.gama
            current_best = self.reef[self.occupied_position[0]]
            reset_count += 1
            if reset_count == self.restart_count:
                self._init_reef__()
                self.reef[0] = self.best_coral
                reset_count = 0
            if current_best['health'] < g_best["health"]:
                g_best = current_best
                reset_count = 0
            self.loss_train.append(g_best["health"])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best["health"]))
        self.solution = [g_best["position"], g_best["health"]]
        return g_best["position"], g_best['health'], self.loss_train
