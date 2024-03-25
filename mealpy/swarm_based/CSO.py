#!/usr/bin/env python
# Created by "Thieu" at 10:09, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalCSO(Optimizer):
    """
    The original version of: Cat Swarm Optimization (CSO)

    Links:
        1. https://link.springer.com/chapter/10.1007/978-3-540-36668-3_94
        2. https://www.hindawi.com/journals/cin/2020/4854895/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + mixture_ratio (float): joining seeking mode with tracing mode, default=0.15
        + smp (int): seeking memory pool, default=5 clones (larger is better but time-consuming)
        + spc (bool): self-position considering, default=False
        + cdc (float): counts of dimension to change (larger is more diversity but slow convergence), default=0.8
        + srd (float): seeking range of the selected dimension (smaller is better but slow convergence), default=0.15
        + c1 (float): same in PSO, default=0.4
        + w_min (float): same in PSO
        + w_max (float): same in PSO
        + selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, CSO
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
    >>> model = CSO.OriginalCSO(epoch=1000, pop_size=50, mixture_ratio = 0.15, smp = 5, spc = False, cdc = 0.8, srd = 0.15, c1 = 0.4, w_min = 0.4, w_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chu, S.C., Tsai, P.W. and Pan, J.S., 2006, August. Cat swarm optimization. In Pacific Rim
    international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, mixture_ratio: float = 0.15, smp: int = 5,
                 spc: bool = False, cdc: float = 0.8, srd: float = 0.15, c1: float = 0.4,
                 w_min: float = 0.5, w_max: float = 0.9, selected_strategy: int = 1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            mixture_ratio (float): joining seeking mode with tracing mode
            smp (int): seeking memory pool, 10 clones  (larger is better but time-consuming)
            spc (bool): self-position considering
            cdc (float): counts of dimension to change  (larger is more diversity but slow convergence)
            srd (float): seeking range of the selected dimension (smaller is better but slow convergence)
            c1 (float): same in PSO
            w_min (float): same in PSO
            w_max (float): same in PSO
            selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mixture_ratio = self.validator.check_float("mixture_ratio", mixture_ratio, (0, 1.0))
        self.smp = self.validator.check_int("smp", smp, [2, 10000])
        self.spc = self.validator.check_bool("spc", spc, [True, False])
        self.cdc = self.validator.check_float("cdc", cdc, (0, 1.0))
        self.srd = self.validator.check_float("srd", srd, (0, 1.0))
        self.c1 = self.validator.check_float("c1", c1, (0, 3.0))
        self.w_min = self.validator.check_float("w_min", w_min, [0.1, 0.5])
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.selected_strategy = self.validator.check_int("selected_strategy", selected_strategy, [0, 4])
        self.set_parameters(["epoch", "pop_size", "mixture_ratio", "smp", "spc", "cdc", "srd", "c1", "w_min", "w_max", "selected_strategy"])
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        """
        + x: current position of cat
        + v: vector v of cat (same amount of dimension as x)
        + flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching) => False: seeking mode , True: tracing mode
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        flag = True if self.generator.uniform() < self.mixture_ratio else False
        return Agent(solution=solution, velocity=velocity, flag=flag)

    def seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = self.generate_population(self.smp)
        if self.spc:
            candidate_cats.append(cat.copy())
            clone_cats = [cat.copy() for _ in range(self.smp - 1)]
        for clone in clone_cats:
            idx = self.generator.choice(range(0, self.problem.n_dims), int(self.cdc * self.problem.n_dims), replace=False)
            pos_new1 = clone.solution * (1 + self.srd)
            pos_new2 = clone.solution * (1 - self.srd)
            pos_new = np.where(self.generator.random(self.problem.n_dims) < 0.5, pos_new1, pos_new2)
            pos_new[idx] = clone.solution[idx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.update(velocity=clone.velocity, flag=clone.flag)
            candidate_cats.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                candidate_cats[-1].target = self.get_target(pos_new)
        candidate_cats = self.update_target_for_population(candidate_cats)

        if self.selected_strategy == 0:  # Best fitness-self
            cat = self.get_best_agent(candidate_cats, self.problem.minmax)
        elif self.selected_strategy == 1:  # Tournament
            k_way = 4
            idx = self.generator.choice(range(0, self.smp), k_way, replace=False)
            cats_k_way = [candidate_cats[_] for _ in idx]
            cat = self.get_best_agent(cats_k_way, self.problem.minmax)
        elif self.selected_strategy == 2:  ### Roul-wheel selection
            list_fitness = [candidate_cats[u].target.fitness for u in range(0, len(candidate_cats))]
            idx = self.get_index_roulette_wheel_selection(list_fitness)
            cat = candidate_cats[idx]
        else:
            idx = self.generator.choice(range(0, len(candidate_cats)))
            cat = candidate_cats[idx]  # Random
        return cat.solution

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            # tracing mode
            if self.pop[idx].flag:
                pos_new = self.pop[idx].solution + w * self.pop[idx].velocity + \
                          self.generator.uniform() * self.c1 * (self.g_best.solution - self.pop[idx].solution)
                pos_new = self.correct_solution(pos_new)
            else:
                pos_new = self.seeking_mode__(self.pop[idx])
            agent.solution = pos_new
            agent.flag = True if self.generator.uniform() < self.mixture_ratio else False
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)
