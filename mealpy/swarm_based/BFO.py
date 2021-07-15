#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

from numpy import sqrt, dot, exp, abs, sum, zeros, append, delete
from numpy.random import uniform, normal
from copy import deepcopy
from mealpy.optimizer import Root


class OriginalBFO(Root):
    """
        The original version of: Bacterial Foraging Optimization (BFO)
        Link:
            + Taken from here: http://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html
    """
    ID_POS = 0
    ID_FIT = 1
    ID_COST = 2
    ID_INTER = 3
    ID_SUM_NUTRIENTS = 4

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 Ci=0.01, Ped=0.25, Ns=4, Ned=5, Nre=50, Nc=10, attract_repesls=(0.1, 0.2, 0.1, 10), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.pop_size = pop_size
        self.step_size = Ci                 # p_eliminate
        self.p_eliminate = Ped              # p_eliminate
        self.swim_length = Ns               # swim_length
        self.elim_disp_steps = Ned          # elim_disp_steps
        self.repro_steps = Nre              # reproduction_steps
        self.chem_steps = Nc                # chem_steps
        self.d_attr = attract_repesls[0]
        self.w_attr = attract_repesls[1]
        self.h_rep = attract_repesls[2]
        self.w_rep = attract_repesls[3]

    def create_solution(self, minmax=None):
        vector = uniform(self.lb, self.ub)
        cost = 0.0
        interaction = 0.0
        fitness = 0.0
        sum_nutrients = 0.0
        return [vector, fitness, cost, interaction, sum_nutrients]

    def _compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem_size * ((cell[self.ID_POS] - other[self.ID_POS]) ** 2).mean(axis=None)
            sum_inter += d * exp(w * diff)
        return sum_inter

    def _attract_repel__(self, cell, cells):
        attract = self._compute_cell_interaction__(cell, cells, -self.d_attr, -self.w_attr)
        repel = self._compute_cell_interaction__(cell, cells, self.h_rep, -self.w_rep)
        return attract + repel

    def _evaluate__(self, cell, cells):
        cell[self.ID_COST] = self.get_fitness_position(cell[self.ID_POS])
        cell[self.ID_INTER] = self._attract_repel__(cell, cells)
        cell[self.ID_FIT] = cell[self.ID_COST] + cell[self.ID_INTER]

    def _tumble_cell__(self, cell, step_size):
        delta_i = uniform(self.lb, self.ub)
        unit_vector = delta_i / sqrt(abs(dot(delta_i, delta_i.T)))
        vector = cell[self.ID_POS] + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def _chemotaxis__(self, l=None, k=None, cells=None):
        current_best = None
        for j in range(0, self.chem_steps):
            moved_cells = []  # New generation
            for i, cell in enumerate(cells):
                sum_nutrients = 0.0
                self._evaluate__(cell, cells)
                if (current_best is None) or cell[self.ID_COST] < current_best[self.ID_COST]:
                    current_best = deepcopy(cell)
                sum_nutrients += cell[self.ID_FIT]

                for m in range(0, self.swim_length):
                    new_cell = self._tumble_cell__(cell, self.step_size)
                    self._evaluate__(new_cell, cells)
                    if current_best[self.ID_COST] > new_cell[self.ID_COST]:
                        current_best = deepcopy(new_cell)
                    if new_cell[self.ID_FIT] > cell[self.ID_FIT]:
                        break
                    cell = deepcopy(new_cell)
                    sum_nutrients += cell[self.ID_FIT]

                cell[self.ID_SUM_NUTRIENTS] = sum_nutrients
                moved_cells.append(deepcopy(cell))
            cells = deepcopy(moved_cells)
            self.loss_train.append(current_best[self.ID_COST])
            if self.verbose:
                print("> Elim: %d, Repro: %d, Chemo: %d, Best fit: %.6f" % (l + 1, k + 1, j + 1, current_best[self.ID_COST]))
        return current_best, cells

    def train(self):
        cells = [self.create_solution() for _ in range(0, self.pop_size)]
        g_best = self.get_global_best_solution(cells, self.ID_FIT, self.ID_MIN_PROB)
        half_pop_size = int(self.pop_size / 2)
        for l in range(0, self.elim_disp_steps):
            for k in range(0, self.repro_steps):
                current_best, cells = self._chemotaxis__(l, k, cells)
                if current_best[self.ID_COST] < g_best[self.ID_COST]:
                    g_best = deepcopy(current_best)
                cells = sorted(cells, key=lambda cell: cell[self.ID_SUM_NUTRIENTS])
                cells = deepcopy(cells[0:half_pop_size]) + deepcopy(cells[0:half_pop_size])
            for idc in range(self.pop_size):
                if uniform() < self.p_eliminate:
                    cells[idc] = self.create_solution()
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class BaseBFO(Root):
    """
    The adaptive version of: Bacterial Foraging Optimization (BFO)
    Notes:
        + This is the best improvement version of BFO
        + Based on this paper: An Adaptive Bacterial Foraging Optimization Algorithm with Lifecycle and Social Learning
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 Ci=(0.1, 0.001), Ped=0.01, Ns=4, N_minmax=(2, 40), **kwargs):
        super().__init__(obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.step_size = Ci  # C_s (start), C_e (end)  -=> step size # step size in BFO
        self.p_eliminate = Ped  # Probability eliminate
        self.swim_length = Ns  # swim_length

        # (Dead threshold value, split threshold value) -> N_adapt, N_split
        self.N_adapt = N_minmax[0]  # Dead threshold value
        self.N_split = N_minmax[1]  # split threshold value

        self.C_s = self.step_size[0] * (self.ub - self.lb)
        self.C_e = self.step_size[1] * (self.ub - self.lb)

    def _tumble_cell__(self, cell=None, cell_local=None, step_size=None, g_best=None, nutrient=None):
        delta_i = (g_best[self.ID_POS] - cell[self.ID_POS]) + (cell_local[self.ID_POS] - cell[self.ID_POS])
        delta = sqrt(abs(dot(delta_i, delta_i.T)))
        if delta == 0:
            unit_vector = uniform(self.lb, self.ub)
        else:
            unit_vector = delta_i / delta
        pos_new = cell[self.ID_POS] + step_size * unit_vector
        pos_new = self.amend_position_faster(pos_new)
        fit_new = self.get_fitness_position(pos_new)
        if fit_new < cell[self.ID_FIT]:
            nutrient += 1
            cell_local = [deepcopy(pos_new), fit_new]  # Update personal best
        else:
            nutrient -= 1
        cell = [deepcopy(pos_new), fit_new]

        # Update global best
        g_best = deepcopy(cell) if g_best[self.ID_FIT] > cell[self.ID_FIT] else g_best
        return cell, cell_local, nutrient, g_best

    def _update_step_size__(self, pop=None, list_nutrient=None, idx=None):
        total_fitness = sum(temp[self.ID_FIT] for temp in pop)
        step_size = self.C_s - (self.C_s - self.C_e) * pop[idx][self.ID_FIT] / total_fitness
        step_size = step_size / list_nutrient[idx] if list_nutrient[idx] > 0 else step_size
        return step_size

    def create_solution(self, minmax=None):
        vector = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(vector)  # Current position
        nutrient = 0
        p_best = deepcopy(vector)
        return [vector, fitness, nutrient, p_best]

    def train(self):
        pop = [self.create_solution() for _ in range(0, self.pop_size)]  # Current population
        g_best = self.get_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        list_nutrient = zeros(self.pop_size)    # total nutrient gained by the bacterium in its whole searching process.(int number)
        pop_local = deepcopy(pop)               # The best population in history

        for epoch in range(self.epoch):
            i = 0
            while i < len(pop):
                step_size = self._update_step_size__(pop, list_nutrient, i)
                JLast = pop[i][self.ID_FIT]
                pop[i], pop_local[i], list_nutrient[i], g_best = self._tumble_cell__(cell=pop[i], cell_local=pop_local[i], step_size=step_size,
                                                                                     g_best=g_best, nutrient=list_nutrient[i])
                m = 0
                while m < self.swim_length:  # Ns
                    if pop[i][self.ID_FIT] < JLast:     # Found better location, move forward to find more nutrient
                        step_size = self._update_step_size__(pop, list_nutrient, i)
                        JLast = pop[i][self.ID_FIT]
                        pop[i], pop_local[i], list_nutrient[i], g_best = self._tumble_cell__(cell=pop[i], cell_local=pop_local[i], step_size=step_size,
                                                                                             g_best=g_best, nutrient=list_nutrient[i])
                        m += 1
                    else:       # Found worst location, stop moving
                        m = self.swim_length

                S_current = len(pop)

                if list_nutrient[i] > max(self.N_split, self.N_split + (S_current - self.pop_size) / self.N_adapt):
                    pos_new = pop[i][self.ID_POS] + normal(self.lb, self.ub, self.problem_size) * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                    fit_new = self.get_fitness_position(pos_new)
                    list_nutrient = append(list_nutrient, [0])
                    pop.append(deepcopy([pos_new, fit_new]))
                    pop_local.append(deepcopy([pos_new, fit_new]))

                if list_nutrient[i] < min(self.N_adapt, self.N_adapt + (S_current - self.pop_size) / self.N_adapt):
                    pop.pop(i)
                    pop_local.pop(i)
                    list_nutrient = delete(list_nutrient, [i])
                    i -= 1

                if uniform() < self.p_eliminate:
                    pop.pop(i)
                    pop_local.pop(i)
                    list_nutrient = delete(list_nutrient, [i])
                    i -= 1
                ## Make sure the population does not have duplicates.
                new_set = set()
                for idx, obj in enumerate(pop):
                    if tuple(obj[self.ID_POS].tolist()) in new_set:
                        pop.pop(idx)
                        pop_local.pop(idx)
                        list_nutrient = delete(list_nutrient, [idx])
                        i -= 1
                    else:
                        new_set.add(tuple(obj[self.ID_POS].tolist()))
                i += 1
            g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Pop_size: {}, Best fit: {}".format(epoch + 1, len(pop), g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
