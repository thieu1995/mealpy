#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy
from copy import deepcopy

from mealpy.utils.agent import Agent


class OriginalDE(Optimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DE
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
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, wf: float = 0.1, cr: float = 0.9, strategy: int = 0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.strategy = self.validator.check_int("strategy", strategy, [0, 5])
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "strategy"])
        self.sort_flag = False

    def mutation__(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.correct_solution(pos_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx_list[0]].solution + self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 1:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.g_best.solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 2:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = self.g_best.solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution) + \
                          self.wf * (self.pop[idx_list[2]].solution - self.pop[idx_list[3]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 3:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = self.pop[idx_list[0]].solution + self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution) + \
                          self.wf * (self.pop[idx_list[3]].solution - self.pop[idx_list[4]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        elif self.strategy == 4:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx].solution + self.wf * (self.g_best.solution - self.pop[idx].solution) + \
                          self.wf * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        else:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx].solution + self.wf * (self.pop[idx_list[0]].solution - self.pop[idx].solution) + \
                          self.wf * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)


class JADE(Optimizer):
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DE
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
    >>> model = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, miu_f: float = 0.5,
                 miu_cr: float = 0.5, pt: float = 0.1, ap: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.pt = self.validator.check_float("pt", pt, (0, 1.0))
        # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr
        self.ap = self.validator.check_float("ap", ap, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "pt", "ap"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = list()

    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects ** 2) / temp

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        temp_f = list()
        temp_cr = list()
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = self.generator.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            temp_f.append(f)
            temp_cr.append(cr)
            top = int(self.pop_size * self.pt)
            x_best = pop_sorted[self.generator.integers(0, top)]
            x_r1 = self.pop[self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[self.generator.integers(0, len(new_pop))]
                if np.any(x_r2.solution - x_r1.solution) and np.any(x_r2.solution - self.pop[idx].solution):
                    break
            x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1.solution - x_r2.solution)
            pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, self.pop[idx].solution)
            j_rand = self.generator.integers(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = self.update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                self.dyn_pop_archive.append(self.pop[idx].copy())
                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])
                self.pop[idx] = pop[idx].copy()
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = deepcopy(archive_pop_new)
        # Update miu_cr and miu_f
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(np.array(list_cr))
        if len(list_f) == 0:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(list_f))


class SADE(Optimizer):
    """
    The original version of: Self-Adaptive Differential Evolution (SADE)

    Links:
        1. https://doi.org/10.1109/CEC.2005.1554904

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DE
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
    >>> model = DE.SADE(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.loop_probability = 50
        self.loop_cr = 5
        self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
        self.crm = 0.5
        self.p1 = 0.5
        self.dyn_list_cr = list()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        list_probability = []
        list_cr = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = self.generator.normal(self.crm, 0.1)
            cr = np.clip(cr, 0, 1)
            list_cr.append(cr)
            while True:
                f = self.generator.normal(0.5, 0.3)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            id1, id2, id3 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            if self.generator.random() < self.p1:
                x_new = self.pop[id1].solution + f * (self.pop[id2].solution - self.pop[id3].solution)
                pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, self.pop[idx].solution)
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.correct_solution(pos_new)
                list_probability.append(True)
            else:
                x_new = self.pop[idx].solution + f * (self.g_best.solution - self.pop[idx].solution) + \
                        f * (self.pop[id1].solution - self.pop[id2].solution)
                pos_new = np.where(self.generator.random(self.problem.n_dims) < cr, x_new, self.pop[idx].solution)
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.correct_solution(pos_new)
                list_probability.append(False)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = self.update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if list_probability[idx]:
                if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                    self.ns1 += 1
                    self.pop[idx] = pop[idx].copy()
                else:
                    self.nf1 += 1
            else:
                if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                    self.ns2 += 1
                    self.dyn_list_cr.append(list_cr[idx])
                    self.pop[idx] = pop[idx].copy()
                else:
                    self.nf2 += 1
        # Update cr and p1
        if epoch / self.loop_cr == 0:
            self.crm = np.mean(self.dyn_list_cr)
            self.dyn_list_cr = list()
        if epoch / self.loop_probability == 0:
            self.p1 = self.ns1 * (self.ns2 + self.nf2) / (self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2))
            self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0


class SAP_DE(Optimizer):
    """
    The original version of: Differential Evolution with Self-Adaptive Populations (SAP_DE)

    Links:
        1. https://doi.org/10.1007/s00500-005-0537-1

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + branch (str): ["ABS" or "REL"], gaussian (absolute) or uniform (relative) method

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, DE
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
    >>> model = DE.SAP_DE(epoch=1000, pop_size=50, branch = "ABS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Teo, J., 2006. Exploring dynamic self-adaptive populations in differential evolution. Soft Computing, 10(8), pp.673-686.
    """

    def __init__(self, epoch: int = 1000, pop_size: int = 100, branch: str = "ABS", **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            branch (str): gaussian (absolute) or uniform (relative) method
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.branch = self.validator.check_str("branch", branch, ["ABS", "REL"])
        self.set_parameters(["epoch", "pop_size", "branch"])
        self.fixed_pop_size = self.pop_size
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        crossover_rate = self.generator.uniform(0, 1)
        mutation_rate = self.generator.uniform(0, 1)
        if self.branch == "ABS":
            pop_size = int(10 * self.problem.n_dims + self.generator.normal(0, 1))
        else:  # elif self.branch == "REL":
            pop_size = int(10 * self.problem.n_dims + self.generator.uniform(-0.5, 0.5))
        return Agent(solution=solution, crossover=crossover_rate, mutation=mutation_rate, pop_size=pop_size)

    def edit_to_range__(self, var=None, lower=0, upper=1, func_value=None):
        while var <= lower or var >= upper:
            if var <= lower:
                var += func_value()
            if var >= upper:
                var -= func_value()
        return var

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            # Choose 3 random element and different to idx
            idxs = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            j = self.generator.integers(0, self.pop_size)
            self.F = self.generator.normal(0, 1)
            ## Crossover
            if self.generator.uniform(0, 1) < self.pop[idx].crossover or idx == j:
                pos_new = self.pop[idxs[0]].solution + self.F * (self.pop[idxs[1]].solution - self.pop[idxs[2]].solution)
                cr_new = self.pop[idxs[0]].crossover + self.F * (self.pop[idxs[1]].crossover - self.pop[idxs[2]].crossover)
                mr_new = self.pop[idxs[0]].mutation + self.F * (self.pop[idxs[1]].mutation - self.pop[idxs[2]].mutation)
                if self.branch == "ABS":
                    ps_new = self.pop[idxs[0]].pop_size + int(self.F * (self.pop[idxs[1]].pop_size - self.pop[idxs[2]].pop_size))
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idxs[0]].pop_size + self.F * (self.pop[idxs[1]].pop_size - self.pop[idxs[2]].pop_size)
                pos_new = self.correct_solution(pos_new)
                cr_new = self.edit_to_range__(cr_new, 0, 1, self.generator.random)
                mr_new = self.edit_to_range__(mr_new, 0, 1, self.generator.random)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    agent.update(crossover=cr_new, mutation=mr_new, pop_size=ps_new)
            else:
                pop.append(self.pop[idx].copy())
            ## Mutation
            if self.generator.uniform(0, 1) < self.pop[idxs[0]].mutation:
                pos_new = self.pop[idx].solution + self.generator.normal(0, self.pop[idxs[0]].mutation)
                cr_new = self.generator.normal(0, 1)
                mr_new = self.generator.normal(0, 1)
                if self.branch == "ABS":
                    ps_new = self.pop[idx].pop_size + int(self.generator.normal(0.5, 1))
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idx].pop_size + self.generator.normal(0, self.pop[idxs[0]].mutation)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    agent.update(crossover=cr_new, mutation=mr_new, pop_size=ps_new)
        pop = self.update_target_for_population(pop)
        # Calculate new population size
        total = np.sum([pop[idx].pop_size for idx in range(0, self.pop_size)])
        if self.branch == "ABS":
            m_new = int(total / self.pop_size)
        else:  # elif self.branch == "REL":
            m_new = int(self.pop_size + total)
        if m_new <= 4:
            m_new = self.fixed_pop_size + int(self.generator.uniform(0, 4))
        elif m_new > 4 * self.fixed_pop_size:
            m_new = self.fixed_pop_size - int(self.generator.uniform(0, 4))
        ## Change population by population size
        if m_new <= self.pop_size:
            self.pop = pop[:m_new]
        else:
            pop_sorted = self.get_sorted_population(pop, self.problem.minmax)
            self.pop = pop + pop_sorted[:m_new - self.pop_size]
        self.pop_size = len(self.pop)
