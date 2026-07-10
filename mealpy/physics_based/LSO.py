# https://github.com/AhmetYP

"""
Light Spectrum Optimizer (LSO) - A Physics-Inspired Metaheuristic Optimization Algorithm

This module implements the Light Spectrum Optimizer (LSO), a physics-inspired
metaheuristic algorithm based on the dispersion of light through a prism.

Original paper:
    Abdel-Basset, M., Mohamed, R.
    Light Spectrum Optimizer: A Novel Physics-Inspired Metaheuristic Optimization Algorithm,
    Mathematics 2022, 10(19), 3466,
    DOI: https://doi.org/10.3390/math10193466
"""

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.special import gammaincinv


class OriginalLSO(Optimizer):
    """
    The original version of: Light Spectrum Optimizer (LSO)

    Links:
        1. https://doi.org/10.3390/math10193466
        2. https://www.mathworks.com/matlabcentral/fileexchange (LSO MATLAB code)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ps (float): [0.01, 0.1], default = 0.05, probability of scattering stages
        + Pe (float): [0.5, 0.8], default = 0.6, controlling parameter to exchange between scattering stages
        + Ph (float): [0.3, 0.6], default = 0.4, probability of hybridization between boundary handling methods
        + B (float): [0.01, 0.1], default = 0.05, exploitation probability in the first scattering stage

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, LSO
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
    >>> model = LSO.OriginalLSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdel-Basset, M., Mohamed, R., 2022. Light Spectrum Optimizer: A Novel Physics-Inspired
    Metaheuristic Optimization Algorithm. Mathematics, 10(19), 3466.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 Ps: float = 0.05, Pe: float = 0.6, Ph: float = 0.4, B: float = 0.05,
                 **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ps (float): probability of first and second scattering stages, default = 0.05
            Pe (float): controlling parameter to exchange between scattering stages, default = 0.6
            Ph (float): probability of hybridization between boundary handling methods, default = 0.4
            B (float): exploitation probability in the first scattering stage, default = 0.05
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.Ps = self.validator.check_float("Ps", Ps, (0, 1.0))
        self.Pe = self.validator.check_float("Pe", Pe, (0, 1.0))
        self.Ph = self.validator.check_float("Ph", Ph, (0, 1.0))
        self.B = self.validator.check_float("B", B, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "Ps", "Pe", "Ph", "B"])
        self.sort_flag = False

        # Fixed parameters from the original paper (Refractive indices for red and violet light)
        self.n_red = 1.3318  # Refractive index for red light (Assumption 2)
        self.n_violet = 1.3435  # Refractive index for violet light (Assumption 2)

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Args:
            vec: Input vector to normalize

        Returns:
            Normalized vector (unit vector)
        """
        norm = np.linalg.norm(vec)
        if norm < self.EPSILON:
            return vec
        return vec / norm

    def _boundary_handling_method1(self, pos_new: np.ndarray) -> np.ndarray:
        """
        First boundary handling method: Clipping to bounds.

        Args:
            pos_new: New position to check

        Returns:
            Position clipped to valid bounds
        """
        U = pos_new > self.problem.ub
        L = pos_new < self.problem.lb
        pos_new = pos_new * (~(U | L)) + self.problem.ub * U + self.problem.lb * L
        return pos_new

    def _boundary_handling_method2(self, pos_new: np.ndarray) -> np.ndarray:
        """
        Second boundary handling method: Random reinitialization for out-of-bound dimensions.

        Args:
            pos_new: New position to check

        Returns:
            Position with reinitalized out-of-bound dimensions
        """
        for j in range(self.problem.n_dims):
            if pos_new[j] > self.problem.ub[j] or pos_new[j] < self.problem.lb[j]:
                pos_new[j] = self.problem.lb[j] + self.generator.random() * \
                             (self.problem.ub[j] - self.problem.lb[j])
        return pos_new

    def _apply_boundary_handling(self, pos_new: np.ndarray) -> np.ndarray:
        """
        Apply boundary handling based on Ph probability.

        Args:
            pos_new: New position to check

        Returns:
            Position within valid bounds
        """
        if self.generator.random() < self.Ph:
            return self._boundary_handling_method1(pos_new)
        else:
            return self._boundary_handling_method2(pos_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Get sorted fitness for scattering calculation
        fitness_list = np.array([agent.target.fitness for agent in self.pop])
        sorted_indices = np.argsort(fitness_list)
        if self.problem.minmax == "max":
            sorted_indices = sorted_indices[::-1]
        sorted_fitness = fitness_list[sorted_indices]

        pop_new = []

        for idx in range(self.pop_size):
            # Select random agent for inner refraction normal
            rand_idx_a = self.generator.integers(0, self.pop_size)
            nA = self.pop[rand_idx_a].solution.copy()

            # Current agent for inner reflection normal
            nB = self.pop[idx].solution.copy()

            # Global best for outer refraction normal
            nC = self.g_best.solution.copy()

            # Mean position of all light rays (incident light)
            pos_list = np.array([agent.solution for agent in self.pop])
            x_bar = np.mean(pos_list, axis=0)

            # Normalized vectors (Equations 6, 7, 8, 11)
            norm_nA = self._normalize_vector(nA)  # Normal vector of inner refraction
            norm_nB = self._normalize_vector(nB)  # Normal vector of inner reflection
            norm_nC = self._normalize_vector(nC)  # Normal vector of outer refraction
            Incid_norm = self._normalize_vector(x_bar)  # Normal vector of incident light

            # Random refractive index between red and violet (Equation 15)
            k = self.n_red + self.generator.random() * (self.n_violet - self.n_red)

            # Calculate L1: Inner refraction (Equation 12)
            dot_nA_inc = np.dot(norm_nA, Incid_norm)
            term1 = (1.0 / k) * (Incid_norm - norm_nA * dot_nA_inc)
            term2_inside = np.abs(1 - (1.0 / k ** 2) + ((1.0 / k ** 2) * dot_nA_inc ** 2))
            term2 = norm_nA * np.sqrt(term2_inside)
            L1 = term1 - term2

            # Calculate L2: Inner reflection (Equation 13)
            dot_L1_nB = np.dot(L1, norm_nB)
            L2 = L1 - 2 * norm_nB * dot_L1_nB

            # Calculate L3: Outer refraction (Equation 14)
            dot_nC_L2 = np.dot(norm_nC, L2)
            term3 = k * (L2 - norm_nC * dot_nC_L2)
            term4_inside = np.abs(1 - k ** 2 + k ** 2 * dot_nC_L2 ** 2)
            term4 = norm_nC * np.sqrt(term4_inside)
            L3 = term3 + term4

            # Adaptive parameter 'a' (Equation 20)
            a = self.generator.random() * (1 - epoch / self.epoch)

            # Compute gamma inverse value (used in Equation 19)
            a_safe = max(a, 1e-10)  # Avoid numerical issues
            try:
                ginv = gammaincinv(1, a_safe)  # gammaincinv(a, 1) in MATLAB = gammaincinv(1, a) in scipy
            except:
                ginv = 0.0

            # GI factor (Equation 19)
            r_rand = self.generator.random()
            if r_rand < 1e-10:
                r_rand = 1e-10
            GI = a * (1.0 / r_rand) * ginv

            # Epsilon vector (Equation 18)
            Epsln = a * self.generator.standard_normal(self.problem.n_dims)

            # Select two random agents for difference vector
            rand_idx1 = self.generator.integers(0, self.pop_size)
            rand_idx2 = self.generator.integers(0, self.pop_size)
            diff_vec = self.pop[rand_idx1].solution - self.pop[rand_idx2].solution

            # Colorful dispersion phase (Equations 17 and 18 in paper)
            p = self.generator.random()
            q = self.generator.random()

            if p <= q:
                # Equation 17
                pos_new = self.pop[idx].solution + GI * Epsln * self.generator.random(self.problem.n_dims) * \
                          (L1 - L3) * diff_vec
            else:
                # Equation 18
                pos_new = self.pop[idx].solution + GI * Epsln * self.generator.random(self.problem.n_dims) * \
                          (L2 - L3) * diff_vec

            # Apply boundary handling
            pos_new = self._apply_boundary_handling(pos_new)

            # Create agent and evaluate
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        # Update population for parallel modes
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Scattering stages
        pop_scatter = []

        for idx in range(self.pop_size):
            # Calculate F factor (Equation 25)
            current_fitness = self.pop[idx].target.fitness
            best_fitness = self.g_best.target.fitness
            worst_fitness = sorted_fitness[-1]

            denom = best_fitness - worst_fitness
            if np.abs(denom) < self.EPSILON:
                F = 0.0
            else:
                F = np.abs((current_fitness - best_fitness) / denom)

            # Decide on scattering stage based on F and Ps
            if F < self.generator.random() or self.generator.random() < self.Ps:
                if self.generator.random() < self.Pe:
                    # First scattering stage (Equation 21)
                    rand_idx1 = self.generator.integers(0, self.pop_size)
                    rand_idx2 = self.generator.integers(0, self.pop_size)

                    diff_scatter = self.pop[rand_idx1].solution - self.pop[rand_idx2].solution
                    exploitation_term = np.zeros(self.problem.n_dims)
                    if self.generator.random() < self.B:
                        exploitation_term = self.generator.random(self.problem.n_dims) * \
                                            (self.g_best.solution - self.pop[idx].solution)

                    pos_new = self.pop[idx].solution + self.generator.random() * diff_scatter + exploitation_term
                else:
                    # Second scattering stage (Equation 22)
                    angle = self.generator.random() * 180
                    pos_new = 2 * np.cos(np.radians(angle)) * (self.g_best.solution * self.pop[idx].solution)
            else:
                # Third scattering stage (Equation 24)
                U_mask = self.generator.random(self.problem.n_dims) > self.generator.random(self.problem.n_dims)

                rand_idx1 = self.generator.integers(0, self.pop_size)
                rand_idx2 = self.generator.integers(0, self.pop_size)
                rand_idx3 = self.generator.integers(0, self.pop_size)

                scatter_component = self.pop[rand_idx1].solution + \
                                    np.abs(self.generator.standard_normal()) * \
                                    (self.pop[rand_idx2].solution - self.pop[rand_idx3].solution)

                pos_new = U_mask * scatter_component + (1 - U_mask) * self.pop[idx].solution

            # Apply boundary handling
            pos_new = self._apply_boundary_handling(pos_new)

            # Create agent
            agent = self.generate_empty_agent(pos_new)
            pop_scatter.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        # Update population for parallel modes
        if self.mode in self.AVAILABLE_MODES:
            pop_scatter = self.update_target_for_population(pop_scatter)
            self.pop = self.greedy_selection_population(self.pop, pop_scatter, self.problem.minmax)


class DevLSO(Optimizer):
    """
    The developed version of: Light Spectrum Optimizer (LSO)

    This version includes some improvements:
        + Uses adaptive parameters that change based on epoch
        + Simplified boundary handling
        + More efficient implementation

    Links:
        1. https://doi.org/10.3390/math10193466

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ps (float): [0.01, 0.1], default = 0.05, probability of scattering stages
        + Pe (float): [0.5, 0.8], default = 0.6, controlling parameter to exchange between scattering stages
        + B (float): [0.01, 0.1], default = 0.05, exploitation probability in the first scattering stage

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, LSO
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
    >>> model = LSO.DevLSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdel-Basset, M., Mohamed, R., 2022. Light Spectrum Optimizer: A Novel Physics-Inspired
    Metaheuristic Optimization Algorithm. Mathematics, 10(19), 3466.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100,
                 Ps: float = 0.05, Pe: float = 0.6, B: float = 0.05,
                 **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ps (float): probability of first and second scattering stages, default = 0.05
            Pe (float): controlling parameter to exchange between scattering stages, default = 0.6
            B (float): exploitation probability in the first scattering stage, default = 0.05
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.Ps = self.validator.check_float("Ps", Ps, (0, 1.0))
        self.Pe = self.validator.check_float("Pe", Pe, (0, 1.0))
        self.B = self.validator.check_float("B", B, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "Ps", "Pe", "B"])
        self.sort_flag = False

        # Fixed parameters from the original paper
        self.n_red = 1.3318
        self.n_violet = 1.3435

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Adaptive parameters
        progress = epoch / self.epoch
        a = self.generator.random() * (1 - progress)

        # Get mean position
        pos_list = np.array([agent.solution for agent in self.pop])
        x_bar = np.mean(pos_list, axis=0)

        # Get sorted fitness
        fitness_list = np.array([agent.target.fitness for agent in self.pop])

        pop_new = []

        for idx in range(self.pop_size):
            # Random refractive index
            k = self.n_red + self.generator.random() * (self.n_violet - self.n_red)

            # Compute light vectors
            current_pos = self.pop[idx].solution
            best_pos = self.g_best.solution

            # Simplified light dispersion computation
            rand_idx = self.generator.integers(0, self.pop_size)
            rand_pos = self.pop[rand_idx].solution

            # Normalized directions
            norm_current = current_pos / (np.linalg.norm(current_pos) + self.EPSILON)
            norm_best = best_pos / (np.linalg.norm(best_pos) + self.EPSILON)
            norm_mean = x_bar / (np.linalg.norm(x_bar) + self.EPSILON)

            # Compute adaptive gamma factor
            try:
                ginv = gammaincinv(1, max(a, 1e-10))
            except:
                ginv = 0.0

            r_rand = max(self.generator.random(), 1e-10)
            GI = a * (1.0 / r_rand) * ginv

            # Random noise vector
            epsilon = a * self.generator.standard_normal(self.problem.n_dims)

            # Light dispersion update (simplified)
            diff_factor = (1 / k) * (norm_mean - norm_current) - k * (norm_best - norm_current)

            rand_idx1 = self.generator.integers(0, self.pop_size)
            rand_idx2 = self.generator.integers(0, self.pop_size)

            pos_new = current_pos + GI * epsilon * self.generator.random(self.problem.n_dims) * \
                      diff_factor * (self.pop[rand_idx1].solution - self.pop[rand_idx2].solution)

            # Boundary correction
            pos_new = self.correct_solution(pos_new)

            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

        # Scattering phase
        best_fit = self.g_best.target.fitness
        worst_fit = np.max(fitness_list) if self.problem.minmax == "min" else np.min(fitness_list)

        pop_scatter = []

        for idx in range(self.pop_size):
            current_fit = self.pop[idx].target.fitness

            denom = best_fit - worst_fit
            if np.abs(denom) < self.EPSILON:
                F = 0.0
            else:
                F = np.abs((current_fit - best_fit) / denom)

            if F < self.generator.random() or self.generator.random() < self.Ps:
                if self.generator.random() < self.Pe:
                    # First scattering
                    r1 = self.generator.integers(0, self.pop_size)
                    r2 = self.generator.integers(0, self.pop_size)

                    pos_new = self.pop[idx].solution + self.generator.random() * \
                              (self.pop[r1].solution - self.pop[r2].solution)

                    if self.generator.random() < self.B:
                        pos_new += self.generator.random(self.problem.n_dims) * \
                                   (self.g_best.solution - self.pop[idx].solution)
                else:
                    # Second scattering
                    angle = self.generator.random() * np.pi
                    pos_new = 2 * np.cos(angle) * self.g_best.solution * self.pop[idx].solution
            else:
                # Third scattering
                mask = self.generator.random(self.problem.n_dims) > 0.5

                r1 = self.generator.integers(0, self.pop_size)
                r2 = self.generator.integers(0, self.pop_size)
                r3 = self.generator.integers(0, self.pop_size)

                scatter = self.pop[r1].solution + np.abs(self.generator.standard_normal()) * \
                          (self.pop[r2].solution - self.pop[r3].solution)

                pos_new = np.where(mask, scatter, self.pop[idx].solution)

            pos_new = self.correct_solution(pos_new)

            agent = self.generate_empty_agent(pos_new)
            pop_scatter.append(agent)

            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)

        if self.mode in self.AVAILABLE_MODES:
            pop_scatter = self.update_target_for_population(pop_scatter)
            self.pop = self.greedy_selection_population(self.pop, pop_scatter, self.problem.minmax)