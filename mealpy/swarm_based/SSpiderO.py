#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalSSpiderO(Optimizer):
    """
    The original version of: Social Spider Optimization (SSpiderO)

    Links:
        1. https://www.hindawi.com/journals/mpe/2018/6843923/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fp_min (float): Female Percent min, default = 0.65
        + fp_max (float): Female Percent max, default = 0.9

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SSpiderO import OriginalSSpiderO
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
    >>> fp_min = 0.65
    >>> fp_max = 0.9
    >>> model = OriginalSSpiderO(epoch, pop_size, fp_min, fp_max)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Luque-Chang, A., Cuevas, E., Fausto, F., Zaldivar, D. and Pérez, M., 2018. Social spider
    optimization algorithm: modifications, applications, and perspectives. Mathematical
    Problems in Engineering, 2018.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_WEI = 2

    def __init__(self, epoch=10000, pop_size=100, fp_min=0.65, fp_max=0.9, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fp_min (float): Female Percent min, default = 0.65
            fp_max (float): Female Percent max, default = 0.9
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        fp_min = self.validator.check_float("fp_min", fp_min, (0., 1.0))
        fp_max = self.validator.check_float("fp_max", fp_max, (0., 1.0))
        self.fp_min, self.fp_max = min((fp_min, fp_max)), max((fp_min, fp_max))
        self.set_parameters(["epoch", "pop_size", "fp_min", "fp_max"])

    def initialization(self):
        fp_temp = self.fp_min + (self.fp_max - self.fp_min) * np.random.uniform()  # Female Aleatory Percent
        self.n_f = int(self.pop_size * fp_temp)  # number of female
        self.n_m = self.pop_size - self.n_f  # number of male
        # Probabilities of attraction or repulsion Proper tuning for better results
        self.p_m = (self.epoch + 1 - np.array(range(1, self.epoch + 1))) / (self.epoch + 1)

        idx_males = np.random.choice(range(0, self.pop_size), self.n_m, replace=False)
        idx_females = set(range(0, self.pop_size)) - set(idx_males)
        if self.pop is None:
            self.pop = self.create_population(self.pop_size)
        self.pop_males = [self.pop[idx] for idx in idx_males]
        self.pop_females = [self.pop[idx] for idx in idx_females]
        self.pop = self.recalculate_weights__(self.pop)

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, weight]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        weight = 0.0
        return [position, target, weight]

    def amend_position(self, position=None, lb=None, ub=None):
        """
        Depend on what kind of problem are we trying to solve, there will be an different amend_position
        function to rebound the position of agent into the valid range.

        Args:
            position: vector position (location) of the solution.
            lb: list of lower bound values
            ub: list of upper bound values

        Returns:
            Amended position (make the position is in bound)
        """
        return np.where(np.logical_and(lb <= position, position <= ub), position, np.random.uniform(lb, ub))

    def move_females__(self, epoch=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        pop = self.pop_females + self.pop_males
        # Start looking for any stronger vibration
        for i in range(0, self.n_f):  # Move the females
            ## Find the position s
            id_min = None
            dist_min = 2 ** 16
            for j in range(0, self.pop_size):
                if self.pop_females[i][self.ID_WEI] < pop[j][self.ID_WEI]:
                    dt = np.linalg.norm(pop[j][self.ID_POS] - self.pop_females[i][self.ID_POS]) / scale_distance
                    if dt < dist_min and dt != 0:
                        dist_min = dt
                        id_min = j
            x_s = np.zeros(self.problem.n_dims)
            vibs = 0
            if id_min is not None:
                vibs = 2 * (pop[id_min][self.ID_WEI] * np.exp(-(np.random.uniform() * dist_min ** 2)))  # Vib for the shortest
                x_s = pop[id_min][self.ID_POS]

            ## Find the position b
            dtb = np.linalg.norm(self.g_best[self.ID_POS] - self.pop_females[i][self.ID_POS]) / scale_distance
            vibb = 2 * (self.g_best[self.ID_WEI] * np.exp(-(np.random.uniform() * dtb ** 2)))

            ## Do attraction or repulsion
            beta = np.random.uniform(0, 1, self.problem.n_dims)
            gamma = np.random.uniform(0, 1, self.problem.n_dims)
            random = 2 * self.p_m[epoch] * (np.random.uniform(0, 1, self.problem.n_dims) - 0.5)
            if np.random.uniform() >= self.p_m[epoch]:  # Do an attraction
                pos_new = self.pop_females[i][self.ID_POS] + vibs * (x_s - self.pop_females[i][self.ID_POS]) * beta + \
                          vibb * (self.g_best[self.ID_POS] - self.pop_females[i][self.ID_POS]) * gamma + random
            else:  # Do a repulsion
                pos_new = self.pop_females[i][self.ID_POS] - vibs * (x_s - self.pop_females[i][self.ID_POS]) * beta - \
                          vibb * (self.g_best[self.ID_POS] - self.pop_females[i][self.ID_POS]) * gamma + random
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            self.pop_females[i][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                self.pop_females[i][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop_females = self.update_target_wrapper_population(self.pop_females)
        self.nfe_epoch += self.n_f

    def move_males__(self, epoch=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        my_median = np.median([it[self.ID_WEI] for it in self.pop_males])
        pop = self.pop_females + self.pop_males
        all_pos = np.array([it[self.ID_POS] for it in pop])
        all_wei = np.array([it[self.ID_WEI] for it in pop]).reshape((self.pop_size, 1))
        total_wei = np.sum(all_wei)
        if total_wei == 0:
            mean = np.mean(all_pos, axis=0)
        else:
            mean = np.sum(all_wei * all_pos, axis=0) / total_wei
        for i in range(0, self.n_m):
            delta = 2 * np.random.uniform(0, 1, self.problem.n_dims) - 0.5
            random = 2 * self.p_m[epoch] * (np.random.random(self.problem.n_dims) - 0.5)

            if self.pop_males[i][self.ID_WEI] >= my_median:  # Spider above the median
                # Start looking for a female with stronger vibration
                id_min = None
                dist_min = 99999999
                for j in range(0, self.n_f):
                    if self.pop_females[j][self.ID_WEI] > self.pop_males[i][self.ID_WEI]:
                        dt = np.linalg.norm(self.pop_females[j][self.ID_POS] - self.pop_males[i][self.ID_POS]) / scale_distance
                        if dt < dist_min and dt != 0:
                            dist_min = dt
                            id_min = j
                x_s = np.zeros(self.problem.n_dims)
                vibs = 0
                if id_min != None:
                    # Vib for the shortest
                    vibs = 2 * (self.pop_females[id_min][self.ID_WEI] * np.exp(-(np.random.uniform() * dist_min ** 2)))
                    x_s = self.pop_females[id_min][self.ID_POS]
                pos_new = self.pop_males[i][self.ID_POS] + vibs * (x_s - self.pop_males[i][self.ID_POS]) * delta + random
            else:
                # Spider below median, go to weighted mean
                pos_new = self.pop_males[i][self.ID_POS] + delta * (mean - self.pop_males[i][self.ID_POS]) + random
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            self.pop_males[i][self.ID_POS] = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                self.pop_males[i][self.ID_TAR] = self.get_target_wrapper(pos_new)
        self.pop_males = self.update_target_wrapper_population(self.pop_males)
        self.nfe_epoch += self.n_m

    ### Crossover
    def crossover__(self, mom=None, dad=None, id=0):
        child1 = np.zeros(self.problem.n_dims)
        child2 = np.zeros(self.problem.n_dims)
        if id == 0:  # arithmetic recombination
            r = np.random.uniform(0.5, 1)  # w1 = w2 when r =0.5
            child1 = np.multiply(r, mom) + np.multiply((1 - r), dad)
            child2 = np.multiply(r, dad) + np.multiply((1 - r), mom)

        elif id == 1:
            id1 = np.random.randint(1, int(self.problem.n_dims / 2))
            id2 = int(id1 + self.problem.n_dims / 2)

            child1[:id1] = mom[:id1]
            child1[id1:id2] = dad[id1:id2]
            child1[id2:] = mom[id2:]

            child2[:id1] = dad[:id1]
            child2[id1:id2] = mom[id1:id2]
            child2[id2:] = dad[id2:]
        elif id == 2:
            temp = int(self.problem.n_dims / 2)
            child1[:temp] = mom[:temp]
            child1[temp:] = dad[temp:]
            child2[:temp] = dad[:temp]
            child2[temp:] = mom[temp:]

        return child1, child2

    def mating__(self):
        # Check whether a spider is good or not (above median)
        my_median = np.median([it[self.ID_WEI] for it in self.pop_males])
        pop_males_new = [self.pop_males[i] for i in range(self.n_m) if self.pop_males[i][self.ID_WEI] > my_median]

        # Calculate the radio
        pop = self.pop_females + self.pop_males
        all_pos = np.array([it[self.ID_POS] for it in pop])
        rad = np.max(all_pos, axis=1) - np.min(all_pos, axis=1)
        r = np.sum(rad) / (2 * self.problem.n_dims)

        # Start looking if there's a good female near
        list_child = []
        couples = []
        for i in range(0, len(pop_males_new)):
            for j in range(0, self.n_f):
                dist = np.linalg.norm(pop_males_new[i][self.ID_POS] - self.pop_females[j][self.ID_POS])
                if dist < r:
                    couples.append([pop_males_new[i], self.pop_females[j]])
        if len(couples) >= 2:
            n_child = len(couples)
            for k in range(n_child):
                child1, child2 = self.crossover__(couples[k][0][self.ID_POS], couples[k][1][self.ID_POS], 0)
                pos1 = self.amend_position(child1, self.problem.lb, self.problem.ub)
                pos2 = self.amend_position(child2, self.problem.lb, self.problem.ub)
                target1 = self.get_target_wrapper(pos1)
                target2 = self.get_target_wrapper(pos2)
                list_child.append([pos1, target1, 0.0])
                list_child.append([pos2, target2, 0.0])

        list_child += self.create_population(self.pop_size - len(list_child))
        self.nfe_epoch += len(list_child)
        return list_child

    def survive__(self, pop=None, pop_child=None):
        n_child = len(pop)
        pop_child = self.get_sorted_strim_population(pop_child, n_child)
        for i in range(0, n_child):
            if self.compare_agent(pop_child[i], pop[i]):
                pop[i] = deepcopy(pop_child[i])
        return pop

    def recalculate_weights__(self, pop=None):
        fit_total, fit_best, fit_worst = self.get_special_fitness(pop)
        for i in range(len(pop)):
            if fit_best == fit_worst:
                pop[i][self.ID_WEI] = np.random.uniform(0.2, 0.8)
            else:
                pop[i][self.ID_WEI] = 0.001 + (pop[i][self.ID_TAR][self.ID_FIT] - fit_worst) / (fit_best - fit_worst)
        return pop

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        self.nfe_epoch = 0
        ### Movement of spiders
        self.move_females__(epoch)
        self.move_males__(epoch)

        # Recalculate weights
        pop = self.pop_females + self.pop_males
        pop = self.recalculate_weights__(pop)

        # Mating Operator
        pop_child = self.mating__()
        pop = self.survive__(pop, pop_child)
        self.pop = self.recalculate_weights__(pop)
        self.nfe_per_epoch = self.nfe_epoch
