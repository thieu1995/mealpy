#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


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
    >>> from mealpy import FloatVar, SSpiderO
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
    >>> model = SSpiderO.OriginalSSpiderO(epoch=1000, pop_size=50, fp_min = 0.65, fp_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Luque-Chang, A., Cuevas, E., Fausto, F., Zaldivar, D. and Pérez, M., 2018. Social spider
    optimization algorithm: modifications, applications, and perspectives. Mathematical Problems in Engineering, 2018.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, fp_min: float = 0.65, fp_max: float = 0.9, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            fp_min (float): Female Percent min, default = 0.65
            fp_max (float): Female Percent max, default = 0.9
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        fp_min = self.validator.check_float("fp_min", fp_min, (0., 1.0))
        fp_max = self.validator.check_float("fp_max", fp_max, (0., 1.0))
        self.fp_min, self.fp_max = min((fp_min, fp_max)), max((fp_min, fp_max))
        self.set_parameters(["epoch", "pop_size", "fp_min", "fp_max"])

    def initialization(self):
        fp_temp = self.fp_min + (self.fp_max - self.fp_min) * self.generator.uniform()  # Female Aleatory Percent
        self.n_f = int(self.pop_size * fp_temp)  # number of female
        self.n_m = self.pop_size - self.n_f  # number of male
        # Probabilities of attraction or repulsion Proper tuning for better results
        self.p_m = (self.epoch + 1 - np.array(range(1, self.epoch + 1))) / (self.epoch + 1)

        idx_males = self.generator.choice(range(0, self.pop_size), self.n_m, replace=False)
        idx_females = set(range(0, self.pop_size)) - set(idx_males)
        if self.pop is None:
            self.pop = self.generate_population(self.pop_size)
        self.pop_males = [self.pop[idx] for idx in idx_males]
        self.pop_females = [self.pop[idx] for idx in idx_females]
        self.pop = self.recalculate_weights__(self.pop)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        weight = 0.0
        return Agent(solution=solution, weight=weight)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rd)

    def move_females__(self, epoch=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        pop = self.pop_females + self.pop_males
        # Start looking for any stronger vibration
        for idx in range(0, self.n_f):  # Move the females
            ## Find the position s
            id_min = None
            dist_min = 2 ** 16
            for jdx in range(0, self.pop_size):
                if self.pop_females[idx].weight < pop[jdx].weight:
                    dt = np.linalg.norm(pop[jdx].solution - self.pop_females[idx].solution) / scale_distance
                    if dt < dist_min and dt != 0:
                        dist_min = dt
                        id_min = jdx
            x_s = np.zeros(self.problem.n_dims)
            vibs = 0
            if id_min is not None:
                vibs = 2 * (pop[id_min].weight * np.exp(-(self.generator.uniform() * dist_min ** 2)))  # Vib for the shortest
                x_s = pop[id_min].solution

            ## Find the position b
            dtb = np.linalg.norm(self.g_best.solution - self.pop_females[idx].solution) / scale_distance
            vibb = 2 * (self.g_best.weight * np.exp(-(self.generator.uniform() * dtb ** 2)))

            ## Do attraction or repulsion
            beta = self.generator.uniform(0, 1, self.problem.n_dims)
            gamma = self.generator.uniform(0, 1, self.problem.n_dims)
            rd_pos = 2 * self.p_m[epoch-1] * (self.generator.uniform(0, 1, self.problem.n_dims) - 0.5)
            if self.generator.uniform() >= self.p_m[epoch-1]:  # Do an attraction
                pos_new = self.pop_females[idx].solution + vibs * (x_s - self.pop_females[idx].solution) * beta + \
                          vibb * (self.g_best.solution - self.pop_females[idx].solution) * gamma + rd_pos
            else:  # Do a repulsion
                pos_new = self.pop_females[idx].solution - vibs * (x_s - self.pop_females[idx].solution) * beta - \
                          vibb * (self.g_best.solution - self.pop_females[idx].solution) * gamma + rd_pos
            pos_new = self.correct_solution(pos_new)
            self.pop_females[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                self.pop_females[idx].target = self.get_target(pos_new)
        self.pop_females = self.update_target_for_population(self.pop_females)

    def move_males__(self, epoch=None):
        scale_distance = np.sum(self.problem.ub - self.problem.lb)
        my_median = np.median([it.weight for it in self.pop_males])
        pop = self.pop_females + self.pop_males
        all_pos = np.array([it.solution for it in pop])
        all_wei = np.array([it.weight for it in pop]).reshape((self.pop_size, 1))
        total_wei = np.sum(all_wei)
        if total_wei == 0:
            mean = np.mean(all_pos, axis=0)
        else:
            mean = np.sum(all_wei * all_pos, axis=0) / total_wei
        for idx in range(0, self.n_m):
            delta = 2 * self.generator.uniform(0, 1, self.problem.n_dims) - 0.5
            rd_pos = 2 * self.p_m[epoch-1] * (self.generator.random(self.problem.n_dims) - 0.5)

            if self.pop_males[idx].weight >= my_median:  # Spider above the median
                # Start looking for a female with stronger vibration
                id_min = None
                dist_min = 99999999
                for jdx in range(0, self.n_f):
                    if self.pop_females[jdx].weight > self.pop_males[idx].weight:
                        dt = np.linalg.norm(self.pop_females[jdx].solution - self.pop_males[idx].solution) / scale_distance
                        if dt < dist_min and dt != 0:
                            dist_min = dt
                            id_min = jdx
                x_s = np.zeros(self.problem.n_dims)
                vibs = 0
                if id_min != None:
                    # Vib for the shortest
                    vibs = 2 * (self.pop_females[id_min].weight * np.exp(-(self.generator.uniform() * dist_min ** 2)))
                    x_s = self.pop_females[id_min].solution
                pos_new = self.pop_males[idx].solution + vibs * (x_s - self.pop_males[idx].solution) * delta + rd_pos
            else:
                # Spider below median, go to weighted mean
                pos_new = self.pop_males[idx].solution + delta * (mean - self.pop_males[idx].solution) + rd_pos
            pos_new = self.correct_solution(pos_new)
            self.pop_males[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                self.pop_males[idx].target = self.get_target(pos_new)
        self.pop_males = self.update_target_for_population(self.pop_males)

    ### Crossover
    def crossover__(self, mom=None, dad=None, id=0):
        child1 = np.zeros(self.problem.n_dims)
        child2 = np.zeros(self.problem.n_dims)
        if id == 0:  # arithmetic recombination
            r = self.generator.uniform(0.5, 1)  # w1 = w2 when r =0.5
            child1 = np.multiply(r, mom) + np.multiply((1 - r), dad)
            child2 = np.multiply(r, dad) + np.multiply((1 - r), mom)

        elif id == 1:
            id1 = self.generator.integers(1, int(self.problem.n_dims / 2))
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
        my_median = np.median([it.weight for it in self.pop_males])
        pop_males_new = [self.pop_males[idx] for idx in range(self.n_m) if self.pop_males[idx].weight > my_median]

        # Calculate the radio
        pop = self.pop_females + self.pop_males
        all_pos = np.array([agent.solution for agent in pop])
        rad = np.max(all_pos, axis=1) - np.min(all_pos, axis=1)
        r = np.sum(rad) / (2 * self.problem.n_dims)

        # Start looking if there's a good female near
        list_child = []
        couples = []
        for idx in range(0, len(pop_males_new)):
            for jdx in range(0, self.n_f):
                dist = np.linalg.norm(pop_males_new[idx].solution - self.pop_females[jdx].solution)
                if dist < r:
                    couples.append([pop_males_new[idx], self.pop_females[jdx]])
        if len(couples) >= 2:
            n_child = len(couples)
            for kdx in range(n_child):
                child1, child2 = self.crossover__(couples[kdx][0].solution, couples[kdx][1].solution, 0)
                pos1 = self.correct_solution(child1)
                pos2 = self.correct_solution(child2)
                agent1 = self.generate_agent(pos1)
                agent2 = self.generate_agent(pos2)
                list_child.append(agent1)
                list_child.append(agent2)
        list_child += self.generate_population(self.pop_size - len(list_child))
        return list_child

    def survive__(self, pop=None, pop_child=None):
        n_child = len(pop)
        pop_child = self.get_sorted_and_trimmed_population(pop_child, n_child, self.problem.minmax)
        for idx in range(0, n_child):
            if self.compare_target(pop_child[idx].target, pop[idx].target, self.problem.minmax):
                pop[idx] = pop_child[idx].copy()
        return pop

    def recalculate_weights__(self, pop=None):
        fit_total, fit_best, fit_worst = self.get_special_fitness(pop, self.problem.minmax)
        for idx in range(len(pop)):
            if fit_best == fit_worst:
                pop[idx].weight = self.generator.uniform(0.2, 0.8)
            else:
                pop[idx].weight = 0.001 + (pop[idx].target.fitness - fit_worst) / (fit_best - fit_worst)
        return pop

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
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
