#!/usr/bin/env python
# Created by "Thieu" at 08:37, 17/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import warnings
import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy
from mealpy.utils.opt_info import OptInfo


class OriginalSHADE(Optimizer):
    """
    The original version of: Success-History Adaptation Differential Evolution (SHADE)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 750.
    pop_size : int
        Number of individuals in the population, in range [5, 10000]. Default is 100.
    miu_f : float
        Initial value of the historical memory for the scaling factor F, in range (0.0, 1.0). Default is 0.5.
    miu_cr : float
        Initial value of the historical memory for the crossover rate CR, in range (0.0, 1.0). Default is 0.5.
    memory_size : int, optional
        Number of entries in the historical parameter memories for F and CR, in range [2, 10000].
        If None, the memory size is set equal to `pop_size`. Default is None.

    Notes
    -----
    This implementation was corrected in version 3.1.0 to align with the original SHADE formulation.
    Its behavior may differ from earlier Mealpy versions. Use `DevSHADE` to reproduce the legacy Mealpy implementation.

    References
    ~~~~~~~~~~
    1. Tanabe, R. and Fukunaga, A., 2013, June.
       Success-history based parameter adaptation for differential evolution.
       In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE. https://doi.org/10.1109/CEC.2013.6557555
    """

    OPT_INFO = OptInfo(name="Success-History Adaptation Differential Evolution", year=2013, difficulty="medium", kind="variant")

    def __init__(self, epoch: int = 750, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5,
                 memory_size: int = None, **kwargs: object, ) -> None:
        """
        Args:
            epoch (int): Maximum number of iterations, default = 750.
            pop_size (int): Number of individuals in the population, default = 100.
            miu_f (float): Initial value of the historical memory for the scaling
                factor F, default = 0.5.
            miu_cr (float): Initial value of the historical memory for the crossover
                rate CR, default = 0.5.
            memory_size (int, optional): Number of entries in the historical memories
                for F and CR. If None, it is set to `pop_size`, default = None.
        """
        warnings.warn(
            "`OriginalSHADE` has been corrected to match the original SHADE "
            "paper (since version 3.1.0) and may produce different results from previous Mealpy versions. "
            "Use `DevSHADE` to preserve the legacy Mealpy behavior.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        if memory_size is None:
            self.memory_size = self.pop_size
        else:
            self.memory_size = self.validator.check_int("memory_size", memory_size, [2, 10000])
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "memory_size", ])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_f = np.full(self.memory_size, self.miu_f, dtype=float)
        self.dyn_miu_cr = np.full(self.memory_size, self.miu_cr, dtype=float)
        self.dyn_pop_archive = []
        self.k_counter = 0

    @staticmethod
    def weighted_arithmetic_mean(values, weights):
        return np.sum(weights * values)

    @staticmethod
    def weighted_lehmer_mean(values, weights):
        denominator = np.sum(weights * values)
        if denominator <= 0:
            return 0.0
        return np.sum(weights * values ** 2) / denominator

    def sample_f(self, memory_value):
        while True:
            f = memory_value + 0.1 * self.generator.standard_cauchy()
            if f > 0:
                return min(f, 1.0)

    def trim_archive(self, max_size):
        if len(self.dyn_pop_archive) <= max_size:
            return
        selected = self.generator.choice(len(self.dyn_pop_archive), size=max_size, replace=False)
        self.dyn_pop_archive = [self.dyn_pop_archive[idx] for idx in selected]

    def is_better(self, fit_new, fit_old):
        if self.problem.minmax == "min":
            return fit_new < fit_old
        return fit_new > fit_old

    def is_better_or_equal(self, fit_new, fit_old):
        if self.problem.minmax == "min":
            return fit_new <= fit_old
        return fit_new >= fit_old

    def evolve(self, epoch):
        n_pop = len(self.pop)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted, _ = self.get_sorted_population(pop_old, self.problem.minmax)
        list_cr = np.empty(n_pop)
        list_f = np.empty(n_pop)
        pop_new = []

        for idx in range(n_pop):
            # Randomly select one historical memory entry.
            memory_idx = self.generator.integers(self.memory_size)
            cr = self.generator.normal(self.dyn_miu_cr[memory_idx], 0.1)
            cr = np.clip(cr, 0.0, 1.0)
            f = self.sample_f(self.dyn_miu_f[memory_idx])
            list_cr[idx] = cr
            list_f[idx] = f

            # SHADE 2013: p_i ~ U(2 / NP, 0.2).
            p_min = min(2.0 / n_pop, 0.2)
            p = self.generator.uniform(p_min, 0.2)
            p_num = max(2, int(n_pop * p + 0.5))
            p_num = min(p_num, n_pop)
            x_pbest = pop_sorted[self.generator.integers(p_num)].solution

            # Select r1 from the population, r1 != i.
            r1_candidates = [j for j in range(n_pop) if j != idx]
            r1_idx = self.generator.choice(r1_candidates)
            # Select r2 from P U A, r2 != i and r2 != r1.
            union_pop = pop_old + self.dyn_pop_archive

            r2_candidates = [j for j in range(len(union_pop)) if j != idx and j != r1_idx]
            r2_idx = self.generator.choice(r2_candidates)
            x_i = pop_old[idx].solution
            x_r1 = pop_old[r1_idx].solution
            x_r2 = union_pop[r2_idx].solution

            # DE/current-to-pbest/1.
            mutant = (x_i + f * (x_pbest - x_i) + f * (x_r1 - x_r2))
            # Binomial crossover with mandatory j_rand.
            condition = (self.generator.random(self.problem.n_dims) < cr)
            j_rand = self.generator.integers(self.problem.n_dims)
            condition[j_rand] = True

            pos_new = np.where(condition, mutant, x_i)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        success_cr = []
        success_f = []
        fitness_improvements = []
        next_pop = []
        for idx in range(n_pop):
            parent = pop_old[idx]
            child = pop_new[idx]

            # Standard DE survivor selection uses <=.
            if self.is_better_or_equal(child.target.fitness, parent.target.fitness):
                next_pop.append(child.copy())
            else:
                next_pop.append(parent.copy())

            # Archive and parameter adaptation use strict improvement.
            if self.is_better(child.target.fitness, parent.target.fitness):
                self.dyn_pop_archive.append(parent.copy())
                success_cr.append(list_cr[idx])
                success_f.append(list_f[idx])
                fitness_improvements.append(abs(parent.target.fitness - child.target.fitness))
        self.pop = next_pop

        # SHADE archive size: |A| <= |P|.
        self.trim_archive(n_pop)
        if len(success_f) > 0:
            success_cr = np.asarray(success_cr)
            success_f = np.asarray(success_f)
            fitness_improvements = np.asarray(fitness_improvements)
            total_improvement = np.sum(fitness_improvements)
            if total_improvement > 0:
                weights = (fitness_improvements / total_improvement)
            else:
                weights = np.full(len(success_f), 1.0 / len(success_f), )
            # SHADE 2013:
            # M_CR uses weighted arithmetic mean.
            self.dyn_miu_cr[self.k_counter] = (self.weighted_arithmetic_mean(success_cr, weights))
            # M_F uses weighted Lehmer mean.
            self.dyn_miu_f[self.k_counter] = (self.weighted_lehmer_mean(success_f, weights))
            self.k_counter += 1
            if self.k_counter >= self.memory_size:
                self.k_counter = 0


class DevSHADE(Optimizer):
    """
    The developed version of: Success-History Adaptation Differential Evolution (SHADE)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 750.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    miu_f : float
        Initial weighting factor, in range (0.0, 1.0). Default is 0.5.
    miu_cr : float
        Initial cross-over probability, in range (0.0, 1.0). Default is 0.5.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SHADE
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = SHADE.DevSHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Success-History Adaptation Differential Evolution (Dev)", year=2026, difficulty="medium", kind="variant")

    def __init__(self, epoch: int = 750, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        # the initial f, location is changed then that f is good
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        # the initial cr,
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.k_counter = 0

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = list_weights * list_objects ** 2
        down = list_weights * list_objects
        return np.sum(up) / np.sum(down)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()
        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted, _ = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = self.generator.integers(0, self.pop_size)
            cr = self.generator.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = self.generator.uniform(2 / self.pop_size, 0.2)
            top = int(self.pop_size * p)
            x_best = pop_sorted[self.generator.integers(0, top)]
            r1_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            new_pop = self.pop + self.dyn_pop_archive
            r2_idx = self.generator.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))
            x_r1 = self.pop[r1_idx].solution
            x_r2 = new_pop[r2_idx].solution
            x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1 - x_r2)
            condition = self.generator.random(self.problem.n_dims) < cr
            pos_new = np.where(condition, x_new, self.pop[idx].solution)
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
                list_cr.append(list_cr_new[idx])
                list_f.append(list_f_new[idx])
                list_f_index.append(idx)
                list_cr_index.append(idx)
                self.pop[idx] = pop[idx].copy()
                self.dyn_pop_archive.append(pop[idx].copy())
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, agent in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(agent.copy())
            self.dyn_pop_archive = archive_pop_new

        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for idx in range(0, self.pop_size):
                if idx in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[idx].target.fitness
                    list_fit_new[idx_increase] = self.pop[idx].target.fitness
                    idx_increase += 1
            temp = np.sum(np.abs(list_fit_new - list_fit_old))
            if temp == 0:
                list_weights = 1.0 / len(list_fit_new) * np.ones(len(list_fit_new))
            else:
                list_weights = np.abs(list_fit_new - list_fit_old) / temp
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.pop_size:
                self.k_counter = 0


class OriginalL_SHADE(Optimizer):
    """
    The original version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (L-SHADE)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 750.
    pop_size : int
        Initial number of individuals in the population, in range [4, 10000]. The population size is progressively
        reduced during the search using linear population size reduction. Default is 100.
    miu_f : float
        Initial value of the historical memory for the scaling factor F, in range (0.0, 1.0). Default is 0.5.
    miu_cr : float
        Initial value of the historical memory for the crossover rate CR, in range (0.0, 1.0). Default is 0.5.
    memory_size : int
        Number of entries in the historical parameter memories for F and CR, in range [2, 1000].
        The original L-SHADE configuration uses 6. Default is 6.
    p : float
        Proportion of top-ranked individuals considered when randomly selecting the p-best individual
        in the current-to-pbest mutation strategy, in range (0.0, 1.0). The original L-SHADE
        configuration uses 0.11. Default is 0.11.
    arc_rate : float
        Archive-size ratio controlling the maximum number of individuals stored in the external archive
        relative to the current population size, in range (0.0, 10.0). The archive capacity is approximately
        `arc_rate * current_population_size`. The original L-SHADE configuration uses 2.6. Default is 2.6.

    References
    ~~~~~~~~~~
    1. Tanabe, R. and Fukunaga, A.S., 2014, July.
       Improving the search performance of SHADE using linear population size reduction.
       In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE.
       https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SHADE
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = SHADE.OriginalL_SHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, memory_size=20, p=0.2, arc_rate=1.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Linear Population Size Reduction Success-History Adaptation Differential Evolution",
                       year=2014, difficulty="hard", kind="variant")

    def __init__(self, epoch: int = 750, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5,
                 memory_size: int = 6, p: float = 0.11, arc_rate: float = 2.6, **kwargs: object, ) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [4, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.memory_size = self.validator.check_int("memory_size", memory_size, [2, 1000])
        self.p = self.validator.check_float("p", p, (0, 1.0))
        self.arc_rate = self.validator.check_float("arc_rate", arc_rate, (0, 10.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "memory_size", "p", "arc_rate"])
        self.sort_flag = False
        self.max_nfe = None

    def initialize_variables(self):
        self.dyn_miu_f = np.full(self.memory_size, self.miu_f, dtype=float)
        self.dyn_miu_cr = np.full(self.memory_size, self.miu_cr, dtype=float)
        self.dyn_pop_archive = []
        self.k_counter = 0
        self.n_init = self.pop_size
        self.n_min = 4
        self.dyn_pop_size = self.pop_size
        if (self.termination is not None and self.termination.max_fe is not None):
            self.max_nfe = self.termination.max_fe

    @staticmethod
    def weighted_lehmer_mean(values, weights):
        denominator = np.sum(weights * values)
        if denominator <= 0:
            return 0.0
        return np.sum(weights * values ** 2) / denominator

    def sample_f(self, memory_value):
        while True:
            f = (memory_value + 0.1 * self.generator.standard_cauchy())
            if f > 0:
                return min(f, 1.0)

    def sample_cr(self, memory_value):
        # NaN represents the terminal value ⊥.
        if np.isnan(memory_value):
            return 0.0
        cr = self.generator.normal(memory_value, 0.1)
        return np.clip(cr, 0.0, 1.0)

    def repair_mutant(self, mutant, parent):
        """
        Apply the parent-midpoint boundary repair defined
        for SHADE 1.1 / L-SHADE.
        """
        mutant = np.where(mutant < self.problem.lb, (self.problem.lb + parent) / 2.0, mutant)
        mutant = np.where(mutant > self.problem.ub, (self.problem.ub + parent) / 2.0, mutant)
        return mutant

    def trim_archive(self, max_size):
        if max_size <= 0:
            self.dyn_pop_archive = []
            return
        if len(self.dyn_pop_archive) <= max_size:
            return
        selected = self.generator.choice(len(self.dyn_pop_archive), size=max_size, replace=False)
        self.dyn_pop_archive = [self.dyn_pop_archive[idx] for idx in selected]

    def is_better(self, fit_new, fit_old):
        if self.problem.minmax == "min":
            return fit_new < fit_old
        return fit_new > fit_old

    def is_better_or_equal(self, fit_new, fit_old):
        if self.problem.minmax == "min":
            return fit_new <= fit_old
        return fit_new >= fit_old

    def get_lpsr_population_size(self, epoch):
        """
        Compute the next population size.
        """
        if self.max_nfe is not None:
            progress = min(self.nfe_counter / self.max_nfe, 1.0)
        else:
            progress = min(epoch / self.epoch, 1.0)
        size = (self.n_init + (self.n_min - self.n_init) * progress)
        size = int(size + 0.5)
        return max(self.n_min, size)

    def evolve(self, epoch):
        n_pop = len(self.pop)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted, _ = self.get_sorted_population(pop_old, self.problem.minmax)
        list_cr = np.empty(n_pop)
        list_f = np.empty(n_pop)
        pop_new = []
        for idx in range(n_pop):
            # Select one historical memory entry.
            memory_idx = self.generator.integers(self.memory_size)
            cr = self.sample_cr(self.dyn_miu_cr[memory_idx])
            f = self.sample_f(self.dyn_miu_f[memory_idx])

            list_cr[idx] = cr
            list_f[idx] = f
            # L-SHADE uses fixed p.
            p_num = int(n_pop * self.p + 0.5)
            p_num = max(2, p_num)
            p_num = min(n_pop, p_num)
            x_pbest = pop_sorted[self.generator.integers(p_num)].solution

            # Select r1 from P, r1 != i.
            r1_candidates = [j for j in range(n_pop) if j != idx]
            r1_idx = self.generator.choice(r1_candidates)

            # Select r2 from P U A,
            # r2 != i and r2 != r1.
            union_pop = pop_old + self.dyn_pop_archive

            r2_candidates = [j for j in range(len(union_pop)) if j != idx and j != r1_idx]
            r2_idx = self.generator.choice(r2_candidates)
            x_i = pop_old[idx].solution
            x_r1 = pop_old[r1_idx].solution
            x_r2 = union_pop[r2_idx].solution

            # DE/current-to-pbest/1.
            mutant = (x_i + f * (x_pbest - x_i) + f * (x_r1 - x_r2))
            # SHADE 1.1 / L-SHADE midpoint boundary repair.
            mutant = self.repair_mutant(mutant, x_i)

            # Binomial crossover with mandatory j_rand.
            condition = (self.generator.random(self.problem.n_dims) < cr)
            j_rand = self.generator.integers(self.problem.n_dims)
            condition[j_rand] = True
            pos_new = np.where(condition, mutant, x_i)
            pos_new = self.problem.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        success_cr = []
        success_f = []
        fitness_improvements = []
        next_pop = []
        for idx in range(n_pop):
            parent = pop_old[idx]
            child = pop_new[idx]

            # Standard DE selection: child survives when f(child) <= f(parent).
            if self.is_better_or_equal(child.target.fitness, parent.target.fitness):
                next_pop.append(child.copy())
            else:
                next_pop.append(parent.copy())

            # Success-history update and archive require - strict improvement.
            if self.is_better(child.target.fitness, parent.target.fitness):
                # Archive the defeated parent, not the winner.
                self.dyn_pop_archive.append(parent.copy())
                success_cr.append(list_cr[idx])
                success_f.append(list_f[idx])
                fitness_improvements.append(abs(parent.target.fitness - child.target.fitness))
        self.pop = next_pop

        # Archive size before LPSR adjustment.
        archive_size = int(self.arc_rate * n_pop + 0.5)
        self.trim_archive(archive_size)
        # Update historical memories.
        if len(success_f) > 0:
            success_cr = np.asarray(success_cr, dtype=float)
            success_f = np.asarray(success_f, dtype=float)
            fitness_improvements = np.asarray(fitness_improvements, dtype=float)
            total_improvement = np.sum(fitness_improvements)
            if total_improvement > 0:
                weights = (fitness_improvements / total_improvement)
            else:
                weights = np.full(len(success_f), 1.0 / len(success_f))

            # SHADE 1.1 terminal-value rule for M_CR.
            if (np.isnan(self.dyn_miu_cr[self.k_counter]) or np.max(success_cr) == 0.0):
                self.dyn_miu_cr[self.k_counter] = np.nan
            else:
                self.dyn_miu_cr[self.k_counter] = self.weighted_lehmer_mean(success_cr, weights)
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(success_f, weights)
            self.k_counter += 1
            if self.k_counter >= self.memory_size:
                self.k_counter = 0
        # Linear Population Size Reduction.
        new_pop_size = self.get_lpsr_population_size(epoch)
        if new_pop_size < len(self.pop):
            self.pop = (self.get_sorted_and_trimmed_population(self.pop, new_pop_size, self.problem.minmax))
        self.dyn_pop_size = len(self.pop)

        # Resize archive according to the reduced population.
        archive_size = int(self.arc_rate * self.dyn_pop_size + 0.5)
        self.trim_archive(archive_size)


class DevL_SHADE(Optimizer):
    """
    The developed version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (L-SHADE)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 750.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    miu_f : float
        Initial weighting factor, in range (0.0, 1.0). Default is 0.5.
    miu_cr : float
        Initial cross-over probability, in range (0.0, 1.0). Default is 0.5.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SHADE
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = SHADE.DevL_SHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    OPT_INFO = OptInfo(name="Linear Population Size Reduction Success-History Adaptation Differential Evolution (Dev)",
                       year=2026, difficulty="hard", kind="variant")

    def __init__(self, epoch: int = 750, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr"])
        self.sort_flag = False

    def initialize_variables(self):
        # Dynamic variable
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.dyn_pop_size = self.pop_size
        self.k_counter = 0
        self.n_min = int(self.pop_size / 5)

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)
        return up / down if down != 0 else 0.5

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()
        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted, _ = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = self.generator.integers(0, self.pop_size)
            cr = self.generator.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = self.generator.uniform(0.15, 0.2)
            top = int(np.ceil(self.dyn_pop_size * p))
            x_best = pop_sorted[self.generator.integers(0, top)]
            r1_idx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            new_pop = self.pop + self.dyn_pop_archive
            r2_idx = self.generator.choice(list(set(range(0, len(new_pop))) - {idx, r1_idx}))
            x_r1 = self.pop[r1_idx].solution
            x_r2 = new_pop[r2_idx].solution
            x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1 - x_r2)
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
                list_cr.append(list_cr_new[idx])
                list_f.append(list_f_new[idx])
                list_f_index.append(idx)
                list_cr_index.append(idx)
                self.pop[idx] = pop[idx].copy()
                self.dyn_pop_archive.append(self.pop[idx].copy())
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, agent in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(agent.copy())
            self.dyn_pop_archive = archive_pop_new
        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for idx in range(0, self.dyn_pop_size):
                if idx in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[idx].target.fitness
                    list_fit_new[idx_increase] = self.pop[idx].target.fitness
                    idx_increase += 1
            total_fit = np.sum(np.abs(list_fit_new - list_fit_old))
            list_weights = 0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0
        # Linear Population Size Reduction
        self.dyn_pop_size = round(self.pop_size + epoch * ((self.n_min - self.pop_size) / self.epoch))


class L_SHADE(DevL_SHADE):
    """
    Deprecated alias of :class:`DevL_SHADE`.

    .. deprecated:: 3.1.0
        `L_SHADE` is deprecated and will be removed in version 4.0.0. Use `DevL_SHADE` to preserve
        the behavior of the previous Mealpy implementation, or `OriginalL_SHADE` for the original L-SHADE algorithm.
    """
    OPT_INFO = DevL_SHADE.OPT_INFO
    DEPRECATED = True

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`L_SHADE` is deprecated and will be removed in version 4.0.0. "
            "Use `DevL_SHADE` to preserve the behavior of the previous Mealpy "
            "implementation, or `OriginalL_SHADE` for the original L-SHADE algorithm.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
