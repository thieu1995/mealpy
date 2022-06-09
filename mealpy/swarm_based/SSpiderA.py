# !/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from mealpy.optimizer import Optimizer


class BaseSSpiderA(Optimizer):
    """
    My modified version of: Social Spider Algorithm (BaseSSpiderA)

    Links:
        1. https://doi.org/10.1016/j.asoc.2015.02.014
        2. https://github.com/James-Yu/SocialSpiderAlgorithm  (Modified this version)

    Notes
    ~~~~~
    + The version on above github is very slow convergence
    + Changes the idea of intensity, which one has better intensity, others will move toward to it

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
        + p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
        + p_m (float): the probability of each value in a dimension mask to be one, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SSpiderA import BaseSSpiderA
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
    >>> r_a = 50
    >>> p_c = 0.5
    >>> p_m = 1.0
    >>> model = BaseSSpiderA(problem_dict1, epoch, pop_size, r_a, p_c, p_m)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] James, J.Q. and Li, V.O., 2015. A social spider algorithm for global optimization.
    Applied soft computing, 30, pp.614-627.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_INT = 2
    ID_TARGET_POS = 3
    ID_PREV_MOVE_VEC = 4
    ID_MASK = 5

    def __init__(self, problem, epoch=10000, pop_size=100, r_a=1, p_c=0.7, p_m=0.1, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
        """
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.r_a = self.validator.check_float("r_a", r_a, (0, 5.0))
        self.p_c = self.validator.check_float("p_c", p_c, (0, 1.0))
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class
        + x: The position of s on the web.
        + train: The fitness of the current position of s
        + target_vibration: The target vibration of s in the previous iteration.
        + intensity_vibration: intensity of vibration
        + movement_vector: The movement that s performed in the previous iteration
        + dimension_mask: The dimension mask 1 that s employed to guide movement in the previous iteration
        + The dimension mask is a 0-1 binary vector of length problem size
        + n_changed: The number of iterations since s has last changed its target vibration. (No need)

        Returns:
            list: wrapper of solution with format [position, target, intensity, target_position, previous_movement_vector, dimension_mask]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        intensity = np.log(1. / (abs(target[self.ID_FIT]) + self.EPSILON) + 1)
        target_position = deepcopy(position)
        previous_movement_vector = np.zeros(self.problem.n_dims)
        dimension_mask = np.zeros(self.problem.n_dims)
        return [position, target, intensity, target_position, previous_movement_vector, dimension_mask]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        all_pos = np.array([it[self.ID_POS] for it in self.pop])  ## Matrix (pop_size, problem_size)
        base_distance = np.mean(np.std(all_pos, axis=0))  ## Number
        dist = cdist(all_pos, all_pos, 'euclidean')

        intensity_source = np.array([it[self.ID_INT] for it in self.pop])
        intensity_attenuation = np.exp(-dist / (base_distance * self.r_a))  ## vector (pop_size)
        intensity_receive = np.dot(np.reshape(intensity_source, (1, self.pop_size)), intensity_attenuation)  ## vector (1, pop_size)
        id_best_intennsity = np.argmax(intensity_receive)

        pop_new = []
        for idx in range(0, self.pop_size):
            agent = deepcopy(self.pop[idx])
            if self.pop[id_best_intennsity][self.ID_INT] > self.pop[idx][self.ID_INT]:
                agent[self.ID_TARGET_POS] = self.pop[id_best_intennsity][self.ID_TARGET_POS]
            if np.random.uniform() > self.p_c:  ## changing mask
                agent[self.ID_MASK] = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.p_m, 0, 1)
            pos_new = np.where(self.pop[idx][self.ID_MASK] == 0, self.pop[idx][self.ID_TARGET_POS],
                               self.pop[np.random.randint(0, self.pop_size)][self.ID_POS])
            ## Perform random walk
            pos_new = self.pop[idx][self.ID_POS] + np.random.normal() * \
                      (self.pop[idx][self.ID_POS] - self.pop[idx][self.ID_PREV_MOVE_VEC]) + \
                      (pos_new - self.pop[idx][self.ID_POS]) * np.random.normal()
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
        pop_new = self.update_target_wrapper_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx][self.ID_PREV_MOVE_VEC] = pop_new[idx][self.ID_POS] - self.pop[idx][self.ID_POS]
                self.pop[idx][self.ID_INT] = np.log(1. / (abs(pop_new[idx][self.ID_TAR][self.ID_FIT]) + self.EPSILON) + 1)
                self.pop[idx][self.ID_POS] = pop_new[idx][self.ID_POS]
                self.pop[idx][self.ID_TAR] = pop_new[idx][self.ID_TAR]
