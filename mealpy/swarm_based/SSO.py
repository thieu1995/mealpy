#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:00, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, randint
from numpy import zeros, exp, median, array, sum, multiply, min, max
from numpy.linalg import norm
from copy import deepcopy
from mealpy.root import Root


class BaseSSO(Root):
    """
    The original version of: Social Spider Optimization (SSO)
        (A Social Spider Algorithm for Global Optimization)
    """
    ID_POS = 0
    ID_FIT = 1
    ID_GEN = 2
    ID_WEI = 3

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, fp=(0.65, 0.9), **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.fp = fp                # (fp_min, fp_max): Female Percent

    def create_solution(self, minmax=0, gender="female"):
        solution = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=solution, minmax=minmax)
        gender = gender
        weight = 0.0
        return [solution, fitness, gender, weight]

    def _move_females__(self, n_f=None, pop_females=None, pop_males=None, g_best=None, pm=None):
        scale_distance = sum(self.ub - self.lb)
        pop = pop_females + pop_males
        # Start looking for any stronger vibration
        for i in range(0, n_f):    # Move the females
            ## Find the position s
            id_min = None
            dist_min = 99999999
            for j in range(0, self.pop_size):
                if pop[j][self.ID_WEI] > pop_females[i][self.ID_WEI]:
                    dt = norm(pop[j][self.ID_POS] - pop_females[i][self.ID_POS]) / scale_distance
                    if dt < dist_min and dt != 0:
                        dist_min = dt
                        id_min = j
            x_s = zeros(self.problem_size)
            vibs = 0
            if not (id_min is None):
                vibs = 2*(pop[id_min][self.ID_WEI]*exp(-(uniform()*dist_min**2)))  # Vib for the shortest
                x_s = pop[id_min][self.ID_POS]

            ## Find the position b
            dtb = norm(g_best[self.ID_POS] - pop_females[i][self.ID_POS]) / scale_distance
            vibb = 2 * (g_best[self.ID_WEI] * exp(-(uniform() * dtb ** 2)))

            ## Do attraction or repulsion
            beta = uniform(0, 1, self.problem_size)
            gamma = uniform(0, 1, self.problem_size)
            random = 2 * pm * (uniform(0, 1, self.problem_size) - 0.5)
            if uniform() >= pm:       # Do an attraction
                temp = pop_females[i][self.ID_POS] + vibs * (x_s - pop_females[i][self.ID_POS]) * beta + \
                    vibb * (g_best[self.ID_POS] - pop_females[i][self.ID_POS]) * gamma + random
            else:                               # Do a repulsion
                temp = pop_females[i][self.ID_POS] - vibs * (x_s - pop_females[i][self.ID_POS]) * beta - \
                       vibb * (g_best[self.ID_POS] - pop_females[i][self.ID_POS]) * gamma + random
            temp = self.amend_position_random_faster(temp)
            fit = self.get_fitness_position(temp)
            pop_females[i][self.ID_POS] = temp
            pop_females[i][self.ID_FIT] = fit
        return pop_females

    def _move_males__(self, n_f=None, n_m=None, pop_females=None, pop_males=None, pm=None):
        scale_distance = sum(self.ub - self.lb)
        my_median = median([it[self.ID_WEI] for it in pop_males])
        pop = pop_females + pop_males
        all_pos = array([it[self.ID_POS] for it in pop])
        all_wei = array([it[self.ID_WEI] for it in pop]).reshape((self.pop_size, 1))
        mean = sum(all_wei * all_pos, axis=0) / sum(all_wei)
        for i in range(0, n_m):
            delta = 2 * uniform(0, 1, self.problem_size) - 0.5
            random = 2 * pm * (uniform(0, 1, self.problem_size) - 0.5)

            if pop_males[i][self.ID_WEI] >= my_median:         # Spider above the median
                # Start looking for a female with stronger vibration
                id_min = None
                dist_min = 99999999
                for j in range(0, n_f):
                    if pop_females[j][self.ID_WEI] > pop_males[i][self.ID_WEI]:
                        dt = norm(pop_females[j][self.ID_POS] - pop_males[i][self.ID_POS]) / scale_distance
                        if dt < dist_min and dt != 0:
                            dist_min = dt
                            id_min = j
                x_s = zeros(self.problem_size)
                vibs = 0
                if id_min != None:
                    vibs = 2 * (pop_females[id_min][self.ID_WEI] * exp(-(uniform() * dist_min ** 2)))      # Vib for the shortest
                    x_s = pop_females[id_min][self.ID_POS]
                temp = pop_males[i][self.ID_POS] + vibs * (x_s - pop_males[i][self.ID_POS])*delta + random
            else:
                # Spider below median, go to weighted mean
                temp = pop_males[i][self.ID_POS] + delta * (mean - pop_males[i][self.ID_POS]) + random
            temp = self.amend_position_random_faster(temp)
            fit = self.get_fitness_position(temp)
            pop_males[i][self.ID_POS] = temp
            pop_males[i][self.ID_FIT] = fit
        return pop_males

    ### Crossover
    def _crossover__(self, mom=None, dad=None, id=0):
        child1 = zeros(self.problem_size)
        child2 = zeros(self.problem_size)
        if id == 0:         # arithmetic recombination
            r = uniform(0.5, 1)             # w1 = w2 when r =0.5
            child1 = multiply(r, mom) + multiply((1 - r), dad)
            child2 = multiply(r, dad) + multiply((1 - r), mom)

        elif id == 1:
            id1 = randint(1, int(self.problem_size / 2))
            id2 = int(id1 + self.problem_size / 2)

            child1[:id1] = mom[:id1]
            child1[id1:id2] = dad[id1:id2]
            child1[id2:] = mom[id2:]

            child2[:id1] = dad[:id1]
            child2[id1:id2] = mom[id1:id2]
            child2[id2:] = dad[id2:]
        elif id == 2:
            temp = int(self.problem_size/2)
            child1[:temp] = mom[:temp]
            child1[temp:] = dad[temp:]
            child2[:temp] = dad[:temp]
            child2[temp:] = mom[temp:]

        return child1, child2

    def _mating__(self, pop_females=None, pop_males=None, n_f=None, n_m=None):
        # Check whether a spider is good or not (above median)
        my_median = median([it[self.ID_WEI] for it in pop_males])
        pop_males_new = [pop_males[i] for i in range(n_m) if pop_males[i][self.ID_WEI] > my_median]

        # Calculate the radio
        pop = pop_females + pop_males
        all_pos = array([it[self.ID_POS] for it in pop])
        rad = max(all_pos, axis=1) - min(all_pos, axis=1)
        r = sum(rad)/(2*self.problem_size)

        # Start looking if there's a good female near
        list_child = []
        couples = []
        for i in range(0, len(pop_males_new)):
            for j in range(0, n_f):
                dist = norm(pop_males_new[i][self.ID_POS] - pop_females[j][self.ID_POS])
                if dist < r:
                    couples.append([pop_males_new[i], pop_females[j]])
        if couples:
            n_child = len(couples)
            for k in range(n_child):
                child1, child2 = self._crossover__(couples[k][0][self.ID_POS], couples[k][1][self.ID_POS], 0)
                fit1 = self.get_fitness_position(child1)
                fit2 = self.get_fitness_position(child2)
                list_child.append([child1, fit1, "", 0.0])
                list_child.append([child2, fit2, "", 0.0])
        return list_child

    def _survive__(self, pop_females=None, pop_males=None, pop_child=None):
        n_child = len(pop_child)
        for i in range(0, n_child):
            pop = pop_females + pop_males
            pop = sorted(pop, key=lambda it: it[self.ID_FIT])
            if pop[self.ID_MAX_PROB][self.ID_FIT] > pop_child[i][self.ID_FIT]:
                gender = pop[self.ID_MAX_PROB][self.ID_GEN]
                pop[self.ID_MAX_PROB] = pop_child[i]
                pop[self.ID_MAX_PROB][self.ID_GEN] = gender
        return pop_females, pop_males

    def _recalculate_weights__(self, pop=None):
        pop = sorted(pop, key=lambda temp: temp[self.ID_FIT])
        fit_best = pop[self.ID_MIN_PROB][self.ID_FIT]
        fit_worst = pop[self.ID_MAX_PROB][self.ID_FIT]
        # This will automatically save weight in pop_males and pop_females because this is python.
        for i in range(self.pop_size):
            pop[i][self.ID_WEI] = 0.001 + (pop[i][self.ID_FIT] - fit_worst) / (fit_best - fit_worst)
        return pop

    def train(self):
        fp = self.fp[0] + (self.fp[1] - self.fp[0]) * uniform()       # Female Aleatory Percent
        n_f = int(self.pop_size * fp)       # number of female
        n_m = self.pop_size - n_f           # number of male
        # Probabilities of attraction or repulsion Proper tuning for better results
        p_m = (self.epoch + 1 - array(range(1, self.epoch + 1))) / (self.epoch + 1)

        pop_males = [self.create_solution(minmax=0, gender="male") for _ in range(n_m)]
        pop_females = [self.create_solution(minmax=0, gender="female") for _ in range(n_f)]
        pop = pop_females + pop_males
        pop = self._recalculate_weights__(pop)
        g_best = deepcopy(pop[self.ID_MIN_PROB])

        # Start the iterations
        for epoch in range(self.epoch):
            ### Movement of spiders
            pop_females = self._move_females__(n_f, pop_females, pop_males, g_best, p_m[epoch])
            pop_males = self._move_males__(n_f, n_m, pop_females, pop_males, p_m[epoch])

            # Recalculate weights
            pop = pop_females + pop_males
            pop = self._recalculate_weights__(pop)

            # Mating Operator
            pop_child = self._mating__(pop_females, pop_males, n_f, n_m)
            pop_females, pop_males = self._survive__(pop_females, pop_males, pop_child)

            pop = pop_females + pop_males
            pop = self._recalculate_weights__(pop)
            current_best = deepcopy(pop[self.ID_MIN_PROB])
            if current_best[self.ID_FIT] < g_best[self.ID_FIT]:
                g_best = deepcopy(current_best)

            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

