#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:21, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform
from numpy import sqrt, dot, exp, abs
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from mealpy.root import Root


class BaseBFO(Root):
    """
    Basic version of Bacterial Foraging Optimization Algorithm: Taken from here
    http://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html
    """
    ID_VECTOR = 0
    ID_COST = 1
    ID_INTER = 2
    ID_FITNESS = 3
    ID_SUM_NUTRIENTS = 4

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 pop_size=50, Ci=0.01, Ped=0.25, Ns=4, Ned=5, Nre=50, Nc=10, attract_repesls=(0.1, 0.2, 0.1, 10)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.pop_size = pop_size
        self.step_size = Ci             # p_eliminate
        self.p_eliminate = Ped          # p_eliminate
        self.swim_length = Ns           # swim_length
        self.elim_disp_steps = Ned      # elim_disp_steps
        self.repro_steps = Nre          # reproduction_steps
        self.chem_steps = Nc            # chem_steps
        self.d_attr = attract_repesls[0]
        self.w_attr = attract_repesls[1]
        self.h_rep = attract_repesls[2]
        self.w_rep = attract_repesls[3]

    def _create_solution__(self, minmax=None):
        vector = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        cost = 0.0
        interaction = 0.0
        fitness = 0.0
        sum_nutrients = 0.0
        return [vector, cost, interaction, fitness, sum_nutrients]

    def _compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem_size * mean_squared_error(cell[self.ID_VECTOR], other[self.ID_VECTOR])
            sum_inter += d * exp(w * diff)
        return sum_inter

    def _attract_repel__(self, cell, cells):
        attract = self._compute_cell_interaction__(cell, cells, -self.d_attr, -self.w_attr)
        repel = self._compute_cell_interaction__(cell, cells, self.h_rep, -self.w_rep)
        return attract + repel

    def _evaluate__(self, cell, cells):
        cell[self.ID_COST] = self._fitness_model__(cell[self.ID_VECTOR])
        cell[self.ID_INTER] = self._attract_repel__(cell, cells)
        cell[self.ID_FITNESS] = cell[self.ID_COST] + cell[self.ID_INTER]

    def _tumble_cell__(self, cell, step_size):
        delta_i = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        unit_vector = delta_i / sqrt(abs(dot(delta_i, delta_i.T)))
        vector = cell[self.ID_VECTOR] + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def _chemotaxis__(self, l=None, k = None, cells=None):
        current_best = None
        for j in range(0, self.chem_steps):
            moved_cells = []                            # New generation
            for i, cell in enumerate(cells):
                sum_nutrients = 0.0
                self._evaluate__(cell, cells)
                if (current_best is None) or cell[self.ID_COST] < current_best[self.ID_COST]:
                    current_best = deepcopy(cell)
                sum_nutrients += cell[self.ID_FITNESS]

                for m in range(0, self.swim_length):
                    new_cell = self._tumble_cell__(cell, self.step_size)
                    self._evaluate__(new_cell, cells)
                    if current_best[self.ID_COST] > new_cell[self.ID_COST]:
                        current_best = deepcopy(new_cell)
                    if new_cell[self.ID_FITNESS] > cell[self.ID_FITNESS]:
                        break
                    cell = deepcopy(new_cell)
                    sum_nutrients += cell[self.ID_FITNESS]

                cell[self.ID_SUM_NUTRIENTS] = sum_nutrients
                moved_cells.append(deepcopy(cell))
            cells = deepcopy(moved_cells)
            self.loss_train.append(current_best[self.ID_COST])
            if self.log:
                print("> Elim: %d, Repro: %d, Chemo: %d, Best fit: %.6f" %(l + 1, k + 1, j + 1, current_best[self.ID_COST]))
        return current_best, cells

    def _train__(self):
        cells = [self._create_solution__(minmax=0) for _ in range(0, self.pop_size)]
        half_pop_size = int(self.pop_size / 2)
        best = None
        for l in range(0, self.elim_disp_steps):
            for k in range(0, self.repro_steps):
                current_best, cells = self._chemotaxis__(l, k, cells)
                if (best is None) or current_best[self.ID_COST] < best[self.ID_COST]:
                    best = current_best
                cells = sorted(cells, key=lambda cell: cell[self.ID_SUM_NUTRIENTS])
                cells = deepcopy(cells[0:half_pop_size]) + deepcopy(cells[0:half_pop_size])
            for idc in range(self.pop_size):
                if uniform() < self.p_eliminate:
                    cells[idc] = self._create_solution__(minmax=0)
        return best[self.ID_VECTOR], best[self.ID_FITNESS],self.loss_train


class ABFOLS(Root):
    """
    ### This is the best improvement version of BFO
    ## Paper: An Adaptive Bacterial Foraging Optimization Algorithm with Lifecycle and Social Learning
    """
    ID_VECTOR = 0
    ID_FITNESS = 1
    ID_NUTRIENT = 2
    ID_PERSONAL_BEST = 3

    NUMBER_CONTROL_RATE = 2

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, Ci=(0.1, 0.001), Ped=0.25, Ns=4, N_minmax=(2, 40)):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size
        self.step_size = Ci             # C_s (start), C_e (end)  -=> step size # step size in BFO
        self.p_eliminate = Ped          # Probability eliminate
        self.swim_length = Ns           # swim_length

        # (Dead threshold value, split threshold value) -> N_adapt, N_split
        self.N_adapt = N_minmax[0]      # Dead threshold value
        self.N_split = N_minmax[1]      # split threshold value

        self.C_s = self.step_size[0] * (self.domain_range[1] - self.domain_range[0])
        self.C_e = self.step_size[1] * (self.domain_range[1] - self.domain_range[0])

    def _create_solution__(self, minmax=None):
        vector = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        fitness = self._fitness_model__(vector, self.ID_MAX_PROB)          # Current position
        nutrient = 0          # total nutrient gained by the bacterium in its whole searching process.(int number)
        p_best = deepcopy(vector)
        return [vector, fitness, nutrient, p_best]

    def _tumble_cell__(self, cell=None, step_size=None, g_best=None):
        delta_i = (g_best[self.ID_VECTOR] - cell[self.ID_VECTOR]) + (cell[self.ID_PERSONAL_BEST] - cell[self.ID_VECTOR])
        delta = sqrt(abs(dot(delta_i, delta_i.T)))
        if delta == 0:
            unit_vector = uniform(self.domain_range[0], self.domain_range[1], self.problem_size)
        else:
            unit_vector = delta_i / delta
        #unit_vector = uniform() * 1.2 * (g_best[self.ID_VECTOR] - cell[self.ID_VECTOR]) + uniform() *1.2* (cell[self.ID_PERSONAL_BEST] - cell[self.ID_VECTOR])
        vec = cell[self.ID_VECTOR] + step_size * unit_vector
        fit = self._fitness_model__(vec, self.ID_MAX_PROB)
        if fit> cell[self.ID_FITNESS]:
            cell[self.ID_NUTRIENT] += 1
        else:
            cell[self.ID_NUTRIENT] -= 1

        if fit > cell[self.ID_FITNESS]:             # Update personal best
            cell[self.ID_PERSONAL_BEST] = deepcopy(vec)
        cell[self.ID_VECTOR] = deepcopy(vec)
        cell[self.ID_FITNESS] = fit

        # Update global best
        g_best = deepcopy(cell) if g_best[self.ID_FITNESS] < cell[self.ID_FITNESS] else g_best
        return cell, g_best


    def _update_step_size__(self, cells=None, id=None):
        total_fitness = sum(temp[self.ID_FITNESS] for temp in cells)
        step_size = self.C_s - (self.C_s - self.C_e) * cells[id][self.ID_FITNESS]/ total_fitness
        step_size = step_size / cells[id][self.ID_NUTRIENT] if cells[id][self.ID_NUTRIENT] > 0 else step_size
        return step_size

    def _train__(self):
        cells = [self._create_solution__(minmax=0) for _ in range(0, self.pop_size)]
        g_best = self._get_global_best__(cells, self.ID_FITNESS, self.ID_MAX_PROB)

        for epoch in range(self.epoch):
            i = 0
            while i < len(cells):
                step_size = self._update_step_size__(cells, i)
                JLast = cells[i][self.ID_FITNESS]
                cells[i], g_best = self._tumble_cell__(cell=cells[i], step_size=step_size, g_best=g_best)
                m = 0
                while m < self.swim_length:             # Ns
                    if cells[i][self.ID_FITNESS] < JLast:
                        step_size = self._update_step_size__(cells, i)
                        JLast = cells[i][self.ID_FITNESS]
                        cells[i], g_best = self._tumble_cell__(cell=cells[i], step_size=step_size, g_best=g_best)
                        m += 1
                    else:
                        m = self.swim_length

                S_current = len(cells)
                #print("======= Current Nutrient: {}".format(cells[i][self.ID_NUTRIENT]))

                if cells[i][self.ID_NUTRIENT] > max(self.N_split, self.N_split + (S_current - self.pop_size) / self.N_adapt):
                    new_cell = deepcopy(cells[i])
                    new_cell[self.ID_NUTRIENT] = 0
                    cells[i][self.ID_NUTRIENT] = 0
                    cells.append(new_cell)
                    break

                if cells[i][self.ID_NUTRIENT] < min(self.NUMBER_CONTROL_RATE, self.NUMBER_CONTROL_RATE + (S_current - self.pop_size) / self.N_adapt):
                    cells.pop(i)
                    i -= 1
                    break

                if cells[i][self.ID_NUTRIENT] < self.NUMBER_CONTROL_RATE and uniform() < self.p_eliminate:
                    temp = self._create_solution__(minmax=0)
                    g_best = deepcopy(temp) if temp[self.ID_FITNESS] > g_best[self.ID_FITNESS] else g_best
                    cells[i] = temp
                i += 1
            self.loss_train.append([1.0 / g_best[self.ID_FITNESS], 1.0 / g_best[self.ID_FITNESS]])
            if self.log:
                print("> Epoch: {}, Pop_size: {}, Best fitness: {}".format(epoch + 1, len(cells), 1.0 / g_best[self.ID_FITNESS]))

        return g_best[self.ID_VECTOR], g_best[self.ID_FITNESS], self.loss_train
