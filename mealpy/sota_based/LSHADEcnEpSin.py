#!/usr/bin/env python
# Created by "Thieu" at 14:44, 16/08/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from scipy.stats import cauchy, norm
import numpy as np
from mealpy.optimizer import Optimizer


class OriginalLSHADEcnEpSin(Optimizer):
    """
    The original version of: Ensemble sinusoidal differential covariance matrix adaptation with Euclidean neighborhood (LSHADEcnEpSin)

    Links:
        1. https://doi.org/10.1109/CEC.2017.7969336

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, LSHADEcnEpSin
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
    >>> model = LSHADEcnEpSin.OriginalLSHADEcnEpSin(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5,
    >>>                                freq = 0.5, memory_size = 5, ps = 0.5, pc = 0.4, pop_size_min = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2017, June). Ensemble sinusoidal differential
    covariance matrix adaptation with Euclidean neighborhood for solving CEC2017 benchmark problems.
    In 2017 IEEE congress on evolutionary computation (CEC) (pp. 372-379). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5,
                 freq: float = 0.5, memory_size: int = 5, ps: float = 0.5, pc: float = 0.4,
                 pop_size_min: int = 10, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): [0.1, 1.0], Initial value for F, default = 0.5
            miu_cr (float): [0.1, 1.0], Initial value for CR, default = 0.5
            freq (float): [0.1, 2.0], Initial frequency for sinusoidal adaptation, default = 0.5
            memory_size (int): [1, 20], Memory size for F and CR, default = 5
            ps (float): [0.1, 1.0], Proportion for neighborhood, default = 0.5
            pc (float): [0.1, 1.0], Probability for covariance matrix crossover, default = 0.4
            pop_size_min (int): [5, 1000], Minimum population size, default = 10
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0.1, 1.0))
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0.1, 1.0))
        self.freq = self.validator.check_float("freq", freq, (0.1, 2.0))
        self.memory_size = self.validator.check_int("memory_size", memory_size, [1, 100])
        self.ps = self.validator.check_float("ps", ps, (0.1, 1.0))
        self.pc = self.validator.check_float("pc", pc, (0.1, 1.0))
        self.pop_size_min = self.validator.check_int("pop_size_min", pop_size_min, [4, 1000])
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr", "freq", "memory_size", "ps", "pc", "pop_size_min"])
        self.sort_flag = False

    def initialize_variables(self):
        self.NP_init = self.pop_size if self.pop_size else 18 * self.problem.n_dims
        self.NP_min = self.pop_size_min  # Minimum population size
        self.NP = self.NP_init      # population size will be updated in each iteration

        # Memory settings
        self.H = self.memory_size  # Memory size
        self.M_F = np.full(self.H, self.miu_f)  # Memory for F
        self.M_CR = np.full(self.H, self.miu_cr)  # Memory for CR
        self.M_freq = np.full(self.H, self.freq)  # Memory for frequency
        self.memory_index = 0

        # Sinusoidal parameters
        self.LP = 10  # Learning period
        self.freq_fixed = self.freq  # Fixed frequency for non-adaptive
        self.epsilon = 0.01  # To avoid null success rates

        # Performance tracking for sinusoidal configurations
        self.ns1_history = []  # Success history for config 1
        self.ns2_history = []  # Success history for config 2
        self.nf1_history = []  # Failure history for config 1
        self.nf2_history = []  # Failure history for config 2

    def before_main_loop(self):
        # Initialize archive with initial population
        self.archive = self.pop.copy()

    def update_sinusoidal_probabilities(self, epoch):
        """Update probabilities for sinusoidal configurations"""
        if epoch <= self.LP:
            return 0.5, 0.5  # Equal probabilities initially

        # Calculate success rates
        start_idx = max(0, epoch - self.LP)

        S1 = S2 = self.epsilon
        if len(self.ns1_history) > start_idx:
            ns1_sum = sum(self.ns1_history[start_idx:epoch])
            nf1_sum = sum(self.nf1_history[start_idx:epoch])
            S1 = (ns1_sum + self.epsilon) / (ns1_sum + nf1_sum + 2 * self.epsilon)

        if len(self.ns2_history) > start_idx:
            ns2_sum = sum(self.ns2_history[start_idx:epoch])
            nf2_sum = sum(self.nf2_history[start_idx:epoch])
            S2 = (ns2_sum + self.epsilon) / (ns2_sum + nf2_sum + 2 * self.epsilon)

        # Calculate probabilities
        total_S = S1 + S2
        p1 = S1 / total_S
        p2 = S2 / total_S
        return p1, p2

    def sinusoidal_adaptation(self, epoch, max_epoch, config_type, freq=None):
        """
        Sinusoidal parameter adaptation
        config_type: 1 for non-adaptive decreasing, 2 for adaptive increasing
        """
        if config_type == 1:
            # Non-adaptive sinusoidal decreasing adjustment (Eq. 4)
            F = 0.5 * np.sin(2 * np.pi * self.freq_fixed * (max_epoch - epoch) / max_epoch) + 0.5
        else:
            # Adaptive sinusoidal increasing adjustment (Eq. 5)
            if freq is None:
                freq = self.freq_fixed
            F = 0.5 * np.sin(2 * np.pi * freq * epoch / max_epoch) + 0.5
        return max(0.1, min(1.0, F))  # Ensure F is in valid range

    def current_to_pbest_mutation(self, idx, F, p=0.1):
        """Current-to-pbest/1 mutation strategy"""
        # Select pbest from top p*NP individuals
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        p_size = max(1, int(p * self.NP))
        pbest_idx = self.generator.choice(range(p_size))

        # Select r1 randomly from population (different from i)
        r1 = self.generator.choice(list(set(range(self.NP)) - {idx}))
        # Select r2 from population + archive
        pop_combined = self.pop + self.archive
        r2 = self.generator.choice(list(set(range(len(pop_combined))) - {idx, r1}))

        # Mutation
        pos_new = self.pop[idx].solution + F * (pop_sorted[pbest_idx].solution - self.pop[idx].solution) + \
                    F * (self.pop[r1].solution - pop_combined[r2].solution)
        # Ensure the new position is within bounds
        pos_new = self.correct_solution(pos_new)
        return pos_new

    def binomial_crossover(self, target, mutant, CR=None):
        """Standard binomial crossover"""
        if CR is None:
            r_idx = self.generator.integers(0, self.H)
            CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
            CR = np.clip(CR, 0, 1)
        trial = np.where(self.generator.uniform(0, 1, self.problem.n_dims) <= CR, mutant, target)
        j_rand = self.generator.integers(0, self.problem.n_dims)
        trial[j_rand] = mutant[j_rand]  # Ensure at least one gene from mutant
        return trial

    def covariance_matrix_crossover(self, target, mutant):
        """Covariance matrix learning with Euclidean neighborhood"""
        # Calculate Euclidean distances to best individual
        list_pos = np.array([agent.solution for agent in self.pop])
        dist = np.linalg.norm(list_pos - self.g_best.solution, axis=1)
        dist_indices = np.argsort(dist)

        # Select neighborhood
        neighborhood_size = max(2, int(self.ps * self.NP))
        neighborhood_indices = dist_indices[:neighborhood_size]
        neighborhood = list_pos[neighborhood_indices]

        # Compute covariance matrix
        if neighborhood.shape[0] > 1:
            cov_matrix = np.cov(neighborhood.T)
            # Ensure positive definite
            cov_matrix += np.eye(self.problem.n_dims) * 1e-8

            try:
                # Eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                B = eigenvectors
                B_T = B.T

                # Transform to eigen coordinate system
                target_prime = B_T @ target
                mutant_prime = B_T @ mutant

                # Generate random CR for this crossover
                r_idx = self.generator.integers(0, self.H)
                CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
                CR = np.clip(CR, 0, 1)

                # Binomial crossover in eigen space
                trial_prime = np.where(self.generator.uniform(0, 1, self.problem.n_dims) <= CR, mutant_prime, target_prime)
                j_rand = self.generator.integers(0, self.problem.n_dims)
                trial_prime[j_rand] = mutant_prime[j_rand]  # Ensure at least one gene from mutant
                # Transform back to original coordinate system
                trial = B @ trial_prime
            except np.linalg.LinAlgError:
                # Fallback to regular binomial crossover
                trial = self.binomial_crossover(target, mutant)
        else:
            # Fallback to regular binomial crossover
            trial = self.binomial_crossover(target, mutant)
        return trial

    def weighted_lehmer_mean(self, S_values, delta_f):
        """Calculate weighted Lehmer mean"""
        if len(S_values) == 0:
            return 0.5
        S_values = np.array(S_values)
        delta_f = np.array(delta_f)
        # Avoid division by zero
        delta_f = np.maximum(delta_f, 1e-10)
        weights = delta_f / np.sum(delta_f)
        numerator = np.sum(weights * S_values ** 2)
        denominator = np.sum(weights * S_values)
        if denominator == 0:
            return 0.5
        return numerator / denominator

    def linear_population_reduction(self, epoch, max_epoch):
        """Linear population size reduction"""
        new_NP = int(self.NP_min + (self.NP_init - self.NP_min) * (max_epoch - epoch) / max_epoch)
        new_NP = max(self.NP_min, new_NP)
        if new_NP < self.NP:
            # Sort population by fitness and keep the best individuals
            _, indices = self.get_sorted_population(self.pop, self.problem.minmax, return_index=True)
            tt = indices[:new_NP]
            self.generator.shuffle(tt)
            self.pop = [self.pop[idx] for idx in tt]
        self.NP = new_NP

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Clear successful parameters for this generation
        S_F = []
        S_CR = []
        delta_f = []
        ns1_current = ns2_current = 0
        nf1_current = nf2_current = 0

        for idx in range(0, self.NP):
            if epoch <= self.epoch // 2:
                # First half: sinusoidal adaptation
                p1, p2 = self.update_sinusoidal_probabilities(epoch)
                if self.generator.random() < p1:
                    # Configuration 1: non-adaptive decreasing
                    F = self.sinusoidal_adaptation(epoch, self.epoch, config_type=1)
                    config_used = 1
                else:
                    # Configuration 2: adaptive increasing
                    r_idx = self.generator.integers(0, self.H)
                    freq = cauchy.rvs(loc=self.M_freq[r_idx], scale=0.1)
                    freq = np.clip(freq, 0.1, 2.0)
                    F = self.sinusoidal_adaptation(epoch, self.epoch, config_type=2, freq=freq)
                    config_used = 2
            else:
                # Second half: standard LSHADE
                r_idx = self.generator.integers(0, self.H)
                F = cauchy.rvs(loc=self.M_F[r_idx], scale=0.1)
                F = np.clip(F, 0.1, 1.0)
                config_used = 0

            # Generate CR
            r_idx = self.generator.integers(0, self.H)
            CR = norm.rvs(loc=self.M_CR[r_idx], scale=0.1)
            CR = np.clip(CR, 0, 1)

            # Mutation
            pos_new = self.current_to_pbest_mutation(idx, F)

            # Crossover
            if self.generator.random() < self.pc:
                pos_new = self.covariance_matrix_crossover(self.pop[idx].solution, pos_new)
            else:
                pos_new = self.binomial_crossover(self.pop[idx].solution, pos_new, CR)

            # Calculate fitness
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)

            # Selection
            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax): # Success
                delta = abs(self.pop[idx].target.fitness - agent.target.fitness)
                S_F.append(F)
                S_CR.append(CR)
                delta_f.append(delta)
                # Track performance for sinusoidal configurations
                if epoch <= self.epoch // 2:
                    if config_used == 1:
                        ns1_current += 1
                    elif config_used == 2:
                        ns2_current += 1
                # Add old individual to archive
                self.archive = self.archive + [self.pop[idx].copy()]
                # Replace with trial
                self.pop[idx] = agent
            else:                           # Failure
                if epoch <= self.epoch // 2:
                    if config_used == 1:
                        nf1_current += 1
                    elif config_used == 2:
                        nf2_current += 1

        # Update performance history
        self.ns1_history.append(ns1_current)
        self.ns2_history.append(ns2_current)
        self.nf1_history.append(nf1_current)
        self.nf2_history.append(nf2_current)

        # Update memory
        if len(S_F) > 0:
            ## Update parameter memory using weighted Lehmer mean
            if len(S_F) > 0:
                self.M_F[self.memory_index] = self.weighted_lehmer_mean(S_F, delta_f)
            if len(S_CR) > 0:
                self.M_CR[self.memory_index] = self.weighted_lehmer_mean(S_CR, delta_f)
            self.memory_index = (self.memory_index + 1) % self.H

        # Linear population size reduction
        self.linear_population_reduction(epoch, self.epoch)

        # Limit archive size
        if len(self.archive) > self.NP:
            # Randomly remove excess individuals from archive
            remove_count = len(self.archive) - self.NP
            remove_indices = self.generator.choice(len(self.archive), remove_count, replace=False)
            keep_indices = list(set(range(len(self.archive))) - set(remove_indices))
            self.archive = [self.archive[idx] for idx in keep_indices]
