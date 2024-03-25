#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalSSpiderA(Optimizer):
    """
    The developed version of: Social Spider Algorithm (OriginalSSpiderA)

    Notes:
        + The version of the algorithm available on the GitHub repository has a slow convergence rate.
        + Changes the idea of intensity, which one has better intensity, others will move toward to it
        + https://doi.org/10.1016/j.asoc.2015.02.014
        + https://github.com/James-Yu/SocialSpiderAlgorithm  (Modified this version)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
        + p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
        + p_m (float): the probability of each value in a dimension mask to be one, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SSpiderA
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = SSpiderA.OriginalSSpiderA(epoch=1000, pop_size=50, r_a = 1.0, p_c = 0.7, p_m = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] James, J.Q. and Li, V.O., 2015. A social spider algorithm for global optimization. Applied soft computing, 30, pp.614-627.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, r_a: float = 1.0, p_c: float = 0.7, p_m: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.r_a = self.validator.check_float("r_a", r_a, (0, 5.0))
        self.p_c = self.validator.check_float("p_c", p_c, (0, 1.0))
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "r_a", "p_c", "p_m"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
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
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        target_solution = solution.copy()
        local_vector = np.zeros(self.problem.n_dims)
        mask = np.zeros(self.problem.n_dims)
        return Agent(solution=solution, target_solution=target_solution, local_vector=local_vector, mask=mask)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        """
        Generate new agent with full information

        Args:
            solution (np.ndarray): The solution
        """
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.intensity = np.log(1. / (np.abs(agent.target.fitness) + self.EPSILON) + 1)
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        all_pos = np.array([agent.solution for agent in self.pop])  ## Matrix (pop_size, problem_size)
        base_distance = np.mean(np.std(all_pos, axis=0))  ## Number
        dist = cdist(all_pos, all_pos, 'euclidean')
        intensity_source = np.array([it.intensity for it in self.pop])
        intensity_attenuation = np.exp(-dist / (base_distance * self.r_a))  ## vector (pop_size)
        intensity_receive = np.dot(np.reshape(intensity_source, (1, self.pop_size)), intensity_attenuation)  ## vector (1, pop_size)
        id_best_intensity = np.argmax(intensity_receive)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if self.pop[id_best_intensity].intensity > self.pop[idx].intensity:
                agent.target_solution = self.pop[id_best_intensity].target_solution
            if self.generator.uniform() > self.p_c:  ## changing mask
                agent.mask = np.where(self.generator.uniform(0, 1, self.problem.n_dims) < self.p_m, 0, 1)
            pos_new = np.where(self.pop[idx].mask == 0, self.pop[idx].target_solution, self.pop[self.generator.integers(0, self.pop_size)].solution)
            ## Perform random walk
            pos_new = self.pop[idx].solution + self.generator.normal() * (self.pop[idx].solution - self.pop[idx].local_vector) + \
                      (pos_new - self.pop[idx].solution) * self.generator.normal()
            agent.solution = self.correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(agent.solution)
                agent.intensity = np.log(1. / (np.abs(agent.target.fitness) + self.EPSILON) + 1)
                pop_new.append(agent)
        pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].local_vector = pop_new[idx].solution - self.pop[idx].solution
                self.pop[idx].intensity = np.log(1. / (np.abs(pop_new[idx].target.fitness) + self.EPSILON) + 1)
                self.pop[idx].solution = pop_new[idx].solution
                self.pop[idx].target = pop_new[idx].target
