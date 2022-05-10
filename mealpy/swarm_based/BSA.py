# !/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class BaseBSA(Optimizer):
    """
    The original version of: Bird Swarm Algorithm (BSA)

    Links:
        1. http://doi.org/10.1080/0952813X.2015.1042530
        2. https://www.mathworks.com/matlabcentral/fileexchange/51256-bird-swarm-algorithm-bsa

    Hyper-parameters should fine tuned in approximate range to get faster convergence toward the global optimum:
        + ff (int): (5, 20), flight frequency - default = 10
        + pff (float): the probability of foraging for food - default = 0.8
        + c_couples (list, tuple): [c1, c2] -> (2.0, 2.0), Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
        + a_couples (list, tuple): [a1, a2] -> (1.5, 1.5), The indirect and direct effect on the birds' vigilance behaviours.
        + fl (float): (0.1, 1.0), The followed coefficient - default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.BSA import BaseBSA
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
    >>> ff = 10
    >>> pff = 0.8
    >>> c_couples = [1.5, 1.5]
    >>> a_couples = [1.0, 1.0]
    >>> fl = 0.5
    >>> model = BaseBSA(problem_dict1, epoch, pop_size, ff, pff, c_couples, a_couples, fl)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Meng, X.B., Gao, X.Z., Lu, L., Liu, Y. and Zhang, H., 2016. A new bio-inspired optimisation
    algorithm: Bird Swarm Algorithm. Journal of Experimental & Theoretical Artificial
    Intelligence, 28(4), pp.673-687.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_LBP = 2  # local best position
    ID_LBF = 3  # local best fitness

    def __init__(self, problem, epoch=10000, pop_size=100,
                 ff=10, pff=0.8, c_couples=(1.5, 1.5), a_couples=(1.0, 1.0), fl=0.5, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ff (int): flight frequency - default = 10
            pff (float): the probability of foraging for food - default = 0.8
            c_couples (list, tuple): [c1, c2]: Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
            a_couples (list, tuple): [a1, a2]: The indirect and direct effect on the birds' vigilance behaviours.
            fl (float): The followed coefficient- default = 0.5
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.ff = self.validator.check_int("ff", ff, [2, int(self.pop_size/2)])
        self.pff = self.validator.check_float("pff", pff, (0, 1.0))
        self.c_couples = self.validator.check_tuple_float("c_couples (cognitive, social)", c_couples, ((0, 3.0), (0, 3.0)))
        self.a_couples = self.validator.check_tuple_float("a_couples (indirect, direct)", a_couples, ((0, 3.0), (0, 3.0)))
        self.fl = self.validator.check_float("fl", fl, (0, 1.0))
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None):
        """
        To get the position, fitness wrapper, target and obj list
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [target, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: target
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Returns:
            list: a solution with format [position, target, local_position, local_fitness]
        """
        position = self.generate_position(lb, ub)
        position = self.amend_position(position, lb, ub)
        target = self.get_target_wrapper(position)
        local_position = deepcopy(position)
        local_fitness = deepcopy(target)
        return [position, target, local_position, local_fitness]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pos_list = np.array([item[self.ID_POS] for item in self.pop])
        fit_list = np.array([item[self.ID_LBF][self.ID_FIT] for item in self.pop])
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)
        nfe_epoch = 0

        if epoch % self.ff != 0:
            pop_new = []
            for i in range(0, self.pop_size):
                agent = deepcopy(self.pop[i])
                prob = np.random.uniform() * 0.2 + self.pff  # The probability of foraging for food
                if np.random.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = self.pop[i][self.ID_POS] + self.c_couples[0] * \
                            np.random.uniform() * (self.pop[i][self.ID_LBP] - self.pop[i][self.ID_POS]) + \
                            self.c_couples[1] * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a_couples[0] * np.exp(-self.pop_size * self.pop[i][self.ID_LBF][self.ID_FIT] / (self.EPSILON + fit_sum))
                    k = np.random.choice(list(set(range(0, self.pop_size)) - {i}))
                    t1 = (fit_list[i] - fit_list[k]) / (abs(fit_list[i] - fit_list[k]) + self.EPSILON)
                    A2 = self.a_couples[1] * np.exp(t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON))
                    x_new = self.pop[i][self.ID_POS] + A1 * np.random.uniform(0, 1) * (pos_mean - self.pop[i][self.ID_POS]) + \
                            A2 * np.random.uniform(-1, 1) * (self.g_best[self.ID_POS] - self.pop[i][self.ID_POS])
                agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                pop_new.append(agent)
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
            nfe_epoch += self.pop_size
        else:
            pop_new = deepcopy(self.pop)
            # Divide the bird swarm into two parts: producers and scroungers.
            min_idx = np.argmin(fit_list)
            max_idx = np.argmax(fit_list)
            choose = 0
            if min_idx < 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 1
            if min_idx > 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 2
            if min_idx < 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 3
            if min_idx > 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 4

            if choose < 3:  # Producing (Equation 5)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    agent = deepcopy(self.pop[i])
                    x_new = self.pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[i][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                    pop_new[i] = agent
                if choose == 1:
                    x_new = self.pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx][self.ID_POS]
                    agent = deepcopy(self.pop[min_idx])
                    agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                    pop_new[min_idx] = agent
                for i in range(0, int(self.pop_size / 2)):
                    if choose == 2 or min_idx != i:
                        agent = deepcopy(self.pop[i])
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0.5 * self.pop_size + 1, self.pop_size)
                        x_new = self.pop[i][self.ID_POS] + (self.pop[idx][self.ID_POS] - self.pop[i][self.ID_POS]) * FL
                        agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                        pop_new[i] = agent
            else:  # Scrounging (Equation 6)
                for i in range(0, int(0.5 * self.pop_size)):
                    agent = deepcopy(self.pop[i])
                    x_new = self.pop[i][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[i][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                    pop_new[i] = agent
                if choose == 4:
                    agent = deepcopy(self.pop[min_idx])
                    x_new = self.pop[min_idx][self.ID_POS] + np.random.uniform(self.problem.lb, self.problem.ub) * self.pop[min_idx][self.ID_POS]
                    agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    if choose == 3 or min_idx != i:
                        agent = deepcopy(self.pop[i])
                        FL = np.random.uniform() * 0.4 + self.fl
                        idx = np.random.randint(0, 0.5 * self.pop_size)
                        x_new = self.pop[i][self.ID_POS] + (self.pop[idx][self.ID_POS] - self.pop[i][self.ID_POS]) * FL
                        agent[self.ID_POS] = self.amend_position(x_new, self.problem.lb, self.problem.ub)
                        pop_new[i] = agent
            pop_new = self.update_target_wrapper_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new)
            nfe_epoch += self.pop_size
        self.nfe_per_epoch = nfe_epoch
