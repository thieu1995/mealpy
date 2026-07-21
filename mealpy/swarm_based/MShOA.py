#!/usr/bin/env python
# Created by "Furkan Gunbaz (gunbaz)
# Email: furkan.gunbaz@gmail.com
# Github: https://github.com/gunbaz
# --------------------------------------------------%
# Updated by "Thieu" on 21/07/2026
# Github: https://github.com/thieu1995
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMShOA(Optimizer):
    """
    The original version of: Mantis Shrimp Optimization Algorithm (MShOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, default = 10000.
    pop_size : int
        Number of population size, default = 100.

    Warnings
    --------
    1. Mathematical formulas and notations in the paper are ambiguous, making direct implementation is hard.
    2. This Python code was translated directly from the author's MATLAB implementation.
    3. Use this algorithm with caution due to the questionable quality of the paper.
    4. The main point of this algorithm is changing the position of global best solution instead of current position.

    Links
    -----
    1. https://doi.org/10.3390/math13091500
    2. https://www.mathworks.com/matlabcentral/fileexchange/180937-mantis-shrimp-optimization-algorithm-mshoa

    References
    ~~~~~~~~~~
    1. Sánchez Cortez, J.A., Peraza Vázquez, H., Peña Delgado, A.F., 2025.
       Mantis Shrimp Optimization Algorithm (MShOA): A Novel Bio-Inspired Optimization Algorithm Based on
       Mantis Shrimp Survival Tactics. Mathematics, 13(9), 1500.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MShOA
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
    >>> model = MShOA.OriginalMShOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.polarization = None

    def initialize_variables(self):
        # 1: Foraging, 2: Attack, 3: Burrow/Defense
        self.polarization = self.generator.integers(1, 4, self.pop_size)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        # Boundary control
        flag4ub = solution > self.problem.ub
        flag4lb = solution < self.problem.lb
        out_of_bounds = flag4ub | flag4lb
        # Replace elements that violate bounds with random values
        pos_new = self.problem.lb + self.generator.random(self.problem.n_dims) * (self.problem.ub - self.problem.lb)
        return np.where(out_of_bounds, pos_new, solution)

    def get_polarization(self, prev_pop, current_pop):
        """
        Calculate the polarization array for the agents based on their positions and the new positions.
        """
        fi = 10
        a = 45 - fi
        b = 45 + fi
        c = 135 - fi
        d = 135 + fi

        prev_pos = np.array([agent.solution for agent in prev_pop])
        curr_pos = np.array([agent.solution for agent in current_pop])

        # MATLAB randi(SearchAgents_no) -> 1 to SearchAgents_no
        k1 = self.generator.integers(1, self.pop_size + 1)
        k2 = k1 + self.generator.integers(1, self.pop_size)

        # circshift in python (shift along rows/axis=0)
        prev_pos = np.roll(prev_pos, k1, axis=0)
        curr_pos = np.roll(curr_pos, k2, axis=0)

        dif_1 = np.zeros(self.pop_size)
        idx_ang1 = np.zeros(self.pop_size, dtype=int)
        dif_2 = np.zeros(self.pop_size)
        idx_ang2 = np.zeros(self.pop_size, dtype=int)

        for i in range(self.pop_size):
            norm_pos = np.linalg.norm(prev_pos[i, :]) + 1e-10
            norm_x = np.linalg.norm(curr_pos[i, :]) + 1e-10
            v1 = prev_pos[i, :] / norm_pos
            v2 = curr_pos[i, :] / norm_x

            # Calculate angle in degrees, clip to avoid NaN from floating point inaccuracies
            angulo = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

            if angulo > 90:
                lin_h = 180 - angulo
                lin_v = angulo - 90
                pol_c = 135 - angulo if angulo <= 135 else angulo - 135
                if c <= angulo <= d:
                    pol_c = 0
            else:
                lin_h = angulo - 0
                lin_v = 90 - angulo
                pol_c = 45 - angulo if angulo <= 45 else angulo - 45
                if a <= angulo <= b:
                    pol_c = 0

            ref = np.array([lin_h, lin_v, pol_c])
            dif_1[i] = np.min(ref)
            # Adding 1 to match MATLAB 1-based indexing (1, 2, or 3)
            idx_ang1[i] = np.argmin(ref) + 1

        for i in range(self.pop_size):
            angulo = self.generator.integers(1, 91)

            if angulo > 90:
                lin_h = 180 - angulo
                lin_v = angulo - 90
                pol_c = 135 - angulo if angulo <= 135 else angulo - 135
                if c <= angulo <= d:
                    pol_c = 0
            else:
                lin_h = angulo - 0
                lin_v = 90 - angulo
                pol_c = 45 - angulo if angulo <= 45 else angulo - 45
                if a <= angulo <= b:
                    pol_c = 0

            ref = np.array([lin_h, lin_v, pol_c])
            dif_2[i] = np.min(ref)
            idx_ang2[i] = np.argmin(ref) + 1

        polarization = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            eyes = min(dif_1[i], dif_2[i])
            if eyes == dif_1[i]:
                light_pol = idx_ang1[i]
            elif eyes == dif_2[i]:
                light_pol = idx_ang2[i]
            else:
                light_pol = 1  # Fallback
            polarization[i] = light_pol

        return polarization

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(self.pop_size):
            if self.polarization[idx] == 1:
                # Strategy 1: Foraging
                r = self.generator.integers(0, self.pop_size)
                while idx == r:
                    r = self.generator.integers(0, self.pop_size)
                D = -1 + 2 * self.generator.random()
                pos_new = (self.g_best.solution - self.pop[idx].solution) + D * (self.pop[r].solution - self.g_best.solution)

            elif self.polarization[idx] == 2:
                # Strategy 2: Attack
                angle_t = 180 + (360 - 180) * self.generator.random()
                pos_new = self.g_best.solution * np.cos(np.radians(angle_t))

            else:
                # Strategy 3: Burrow, Defense, or Shelter
                k = 0.3 * self.generator.random()
                bin_val = self.generator.choice([-1, 1])
                pos_new = self.g_best.solution + self.generator.random() * bin_val * k * self.g_best.solution
                if bin_val == 1:
                    angle_t = 180 + (360 - 180) * self.generator.random()
                    pos_new = pos_new * np.cos(np.radians(angle_t))

            # Boundary control
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Update Polarization
        self.polarization = self.get_polarization(self.pop, pop_new)
        self.pop = pop_new


class DevMShOA(Optimizer):
    """
    Our developed version of: Mantis Shrimp Optimization Algorithm (MShOA)

    Parameters
    ----------
    epoch : int
        Maximum number of iterations, in range [1, 100000]. Default is 10000.
    pop_size : int
        Number of population size, in range [5, 10000]. Default is 100.
    k_value : float
        Upper bound for k parameter in defense/shelter phase (Strategy 3, Equation 15).
        k is sampled from U(0, k_value), in range (0.0, 10000.0). Default is 0.3.

    Note
    ~~~~
    1. This version is implemented by "Gunbaz" with the help of AI-generated code.
    2. This implementation uses PTI-based strategy selection exactly as described in Algorithm 1 and Algorithm 2.
    All equations match the paper exactly:
       - Algorithm 1: PTI update mechanism (Eq. 5, 6, 7)
       - Strategy 1: Foraging equation (Eq. 12)
       - Strategy 2: Attack/Strike equation (Eq. 14)
       - Strategy 3: Defense/Burrow equation (Eq. 15)
    3. Each agent has a PTI (Polarization Type Indicator) value ∈ {1, 2, 3} that determines strategy:
       - PTI = 1: Foraging/Navigation (vertical linear polarized light detection) → Strategy 1
       - PTI = 2: Attack/Strike (horizontal linear polarized light detection) → Strategy 2
       - PTI = 3: Defense/Burrow (circular polarized light detection) → Strategy 3

    Links
    -----
    1. https://doi.org/10.3390/math13091500
    2. https://www.mathworks.com/matlabcentral/fileexchange/180937-mantis-shrimp-optimization-algorithm-mshoa

    References
    ~~~~~~~~~~
    1. Sánchez Cortez, J.A., Peraza Vázquez, H., Peña Delgado, A.F., 2025.
       Mantis Shrimp Optimization Algorithm (MShOA): A Novel Bio-Inspired Optimization Algorithm Based on
       Mantis Shrimp Survival Tactics. Mathematics, 13(9), 1500.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MShOA
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
    >>> model = MShOA.DevMShOA(epoch=1000, pop_size=50, k_value=0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, k_value: float = 0.3, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            k_value: Upper bound for k parameter in defense/shelter phase (Strategy 3, Equation 15). 
                    k is sampled from U(0, k_value). Default = 0.3 (matches paper value).
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.k_value = self.validator.check_float("k_value", k_value, (0.0, 10000.))
        self.set_parameters(["epoch", "pop_size", "k_value"])
        self.sort_flag = False
        
        # PTI (Polarization Type Indicator) vector: one value per agent ∈ {1, 2, 3}
        # Initialized randomly according to Algorithm 1 in the paper
        # PTI = 1: Foraging/Navigation (vertical linear polarized light)
        # PTI = 2: Attack/Strike (horizontal linear polarized light)
        # PTI = 3: Defense/Burrow (circular polarized light)
        self.pti = None  # Will be initialized in before_main_loop

    def before_main_loop(self):
        """
        Initialize PTI vector randomly (Algorithm 1, initialization step)
        PTI ∈ {1, 2, 3} for each agent using PTI_i = round(1 + 2 * rand_i)
        This produces distribution: ~25% for 1, ~50% for 2, ~25% for 3
        """
        # Initialize PTI according to paper: PTI_i = round(1 + 2 * rand_i)
        u = self.generator.random(self.pop_size)  # uniform(0, 1) for each agent
        pti_raw = 1 + 2 * u  # produces values in [1, 3)
        self.pti = np.round(pti_raw).astype(int)  # round to nearest integer
        self.pti = np.clip(self.pti, 1, 3)  # ensure values are in {1, 2, 3}

    def evolve(self, epoch: int) -> None:
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class
        Implements Algorithm 2 from the paper with PTI-based strategy selection.
        
        Execution order (critical for correct LPA calculation):
        1. Save X_i(t) (current positions before strategy application)
        2. Apply strategies based on PTI to generate X'_i(t) (new positions)
        3. Calculate LPA from X_i(t) and X'_i(t) (intra-iteration change)
        4. Calculate RPA, LPT, RPT, LAD, RAD
        5. Update PTI according to Algorithm 1

        Args:
            epoch (int): The current iteration
        """
        # Step 1: Extract current positions X_i(t) before strategy application
        pop_pos = np.array([agent.solution for agent in self.pop])  # X_i(t)
        g_best_pos = self.g_best.solution  # Shape: (n_dims,)
        
        # Initialize position update matrix (will become X'_i(t) after strategies)
        pos_new = pop_pos.copy()
        
        # Generate random indices for Strategy 1 (Foraging) - ensure r ≠ i
        random_indices = self.generator.integers(0, self.pop_size, self.pop_size)
        # Ensure r ≠ i: if random_indices[i] == i, replace with another random index (excluding i)
        for idx in range(self.pop_size):
            if random_indices[idx] == idx:
                # Generate random index from [0, pop_size) excluding idx
                candidates = list(range(0, idx)) + list(range(idx + 1, self.pop_size))
                if len(candidates) > 0:
                    random_indices[idx] = self.generator.choice(candidates)
        
        # Create masks for each strategy based on current PTI values
        mask_strategy1 = (self.pti == 1)  # Foraging/Navigation
        mask_strategy2 = (self.pti == 2)  # Attack/Strike
        mask_strategy3 = (self.pti == 3)  # Defense/Burrow
        
        # Expand masks to match dimensions: (pop_size,) -> (pop_size, n_dims)
        mask_s1_expanded = mask_strategy1[:, np.newaxis]
        mask_s2_expanded = mask_strategy2[:, np.newaxis]
        mask_s3_expanded = mask_strategy3[:, np.newaxis]
        
        # Step 2: Apply strategies based on PTI to generate X'_i(t)
        
        # Strategy 1: Foraging/Navigation (PTI = 1) - Equation 12 (Langevin/Brownian)
        # x_i(t+1) = x_best - (x_i(t) - x_best) + D_i(x_r(t) - x_i(t))
        # where D_i: scalar diffusion coefficient per agent, sampled from U(-1, 1) as in Eq. 12
        # and r ≠ i
        random_pop_pos = pop_pos[random_indices]
        v = pop_pos - g_best_pos  # velocity term: (x_i(t) - x_best)
        R_t = random_pop_pos - pop_pos  # random force: (x_r(t) - x_i(t))
        D = self.generator.uniform(-1.0, 1.0, size=(self.pop_size, 1))  # scalar diffusion coefficient per agent
        foraging_pos = g_best_pos - v + D * R_t  # D broadcasts to all dimensions
        pos_new = np.where(mask_s1_expanded, foraging_pos, pos_new)
        
        # Strategy 2: Attack/Strike (PTI = 2) - Equation 14 (circular motion)
        # x_i(t+1) = x_best * cos(θ)
        # where θ ~ U(π, 2π)
        theta = self.generator.uniform(np.pi, 2 * np.pi, size=self.pop_size)[:, np.newaxis]
        strike_pos = g_best_pos * np.cos(theta)  # element-wise multiplication
        pos_new = np.where(mask_s2_expanded, strike_pos, pos_new)
        
        # Strategy 3: Defense/Burrow/Shelter (PTI = 3) - Equation 15
        # Defense: x_i(t+1) = x_best + k * x_best
        # Shelter: x_i(t+1) = x_best - k * x_best
        # where k ~ U(0, k_value)
        # Note: The paper does not explicitly specify the probability distribution for choosing
        # between Defense and Shelter behaviors. This implementation uses uniform (50-50) selection,
        # which is consistent with the paper's description but clarifies an unspecified aspect.
        k = self.generator.uniform(0.0, self.k_value, size=(self.pop_size, 1))  # k ~ U(0, k_value)
        defense_or_shelter = self.generator.random(self.pop_size) < 0.5  # 50% defense, 50% shelter
        defense_or_shelter_expanded = defense_or_shelter[:, np.newaxis]
        # Defense: x_i(t+1) = x_best + k * x_best
        # Shelter: x_i(t+1) = x_best - k * x_best
        defense_pos = np.where(defense_or_shelter_expanded,
                               g_best_pos + k * g_best_pos,  # defense
                               g_best_pos - k * g_best_pos)  # shelter
        pos_new = np.where(mask_s3_expanded, defense_pos, pos_new)
        
        # Step 3: Calculate LPA from intra-iteration change (X_i(t) vs X'_i(t))
        # LPA_i = arccos((X_i(t) · X'_i(t)) / (||X_i(t)|| ||X'_i(t)||))
        # Normalize vectors for dot product calculation
        pop_pos_norm = pop_pos / (np.linalg.norm(pop_pos, axis=1, keepdims=True) + 1e-10)
        pos_new_norm = pos_new / (np.linalg.norm(pos_new, axis=1, keepdims=True) + 1e-10)
        dot_product = np.sum(pop_pos_norm * pos_new_norm, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for arccos
        lpa = np.arccos(dot_product)  # Left Polarization Angle ∈ [0, π]
        
        # Step 4: Calculate RPA, LPT, RPT, LAD, RAD (Algorithm 1)
        
        # Calculate Right Polarization Angle (RPA): RPA_i = rand * π (Eq. 4)
        rpa = self.generator.random(self.pop_size) * np.pi  # RPA ∈ [0, π]
        
        # Determine Left Polarization Type (LPT) and Right Polarization Type (RPT) based on Eq. 5
        # Eq. 5 defines three types with π/8 intervals:
        # Type 1: 3π/8 ≤ açı ≤ 5π/8
        # Type 2: 0 ≤ açı ≤ π/8 or 7π/8 ≤ açı ≤ π
        # Type 3: π/8 < açı < 3π/8 or 5π/8 < açı < 7π/8
        pi_8 = np.pi / 8
        pi_38 = 3 * np.pi / 8
        pi_58 = 5 * np.pi / 8
        pi_78 = 7 * np.pi / 8
        
        # LPT determination from LPA (Eq. 5)
        lpt = np.where(
            (lpa >= pi_38) & (lpa <= pi_58), 1,  # Type 1: 3π/8 ≤ LPA ≤ 5π/8
            np.where(
                ((lpa >= 0) & (lpa <= pi_8)) | ((lpa >= pi_78) & (lpa <= np.pi)), 2,  # Type 2: 0 ≤ LPA ≤ π/8 or 7π/8 ≤ LPA ≤ π
                3  # Type 3: π/8 < LPA < 3π/8 or 5π/8 < LPA < 7π/8
            )
        )
        
        # RPT determination from RPA (Eq. 5)
        rpt = np.where(
            (rpa >= pi_38) & (rpa <= pi_58), 1,  # Type 1: 3π/8 ≤ RPA ≤ 5π/8
            np.where(
                ((rpa >= 0) & (rpa <= pi_8)) | ((rpa >= pi_78) & (rpa <= np.pi)), 2,  # Type 2: 0 ≤ RPA ≤ π/8 or 7π/8 ≤ RPA ≤ π
                3  # Type 3: π/8 < RPA < 3π/8 or 5π/8 < RPA < 7π/8
            )
        )
        
        # Calculate Left Angular Difference (LAD) and Right Angular Difference (RAD) (Eq. 6)
        # Eq. 6 defines piecewise calculation:
        # 0 ≤ açı ≤ π/8 → LAD/RAD = açı
        # 7π/8 ≤ açı ≤ π → LAD/RAD = π − açı
        # 3π/8 ≤ açı ≤ 5π/8 → LAD/RAD = |π/2 − açı|
        # π/8 < açı < 3π/8 → LAD/RAD = |π/4 − açı|
        # 5π/8 < açı < 7π/8 → LAD/RAD = |3π/4 − açı|
        
        # LAD calculation from LPA (Eq. 6)
        lad = np.where(
            (lpa >= 0) & (lpa <= pi_8), lpa,  # 0 ≤ LPA ≤ π/8 → LAD = LPA
            np.where(
                (lpa >= pi_78) & (lpa <= np.pi), np.pi - lpa,  # 7π/8 ≤ LPA ≤ π → LAD = π − LPA
                np.where(
                    (lpa >= pi_38) & (lpa <= pi_58), np.abs(np.pi / 2 - lpa),  # 3π/8 ≤ LPA ≤ 5π/8 → LAD = |π/2 − LPA|
                    np.where(
                        (lpa > pi_8) & (lpa < pi_38), np.abs(np.pi / 4 - lpa),  # π/8 < LPA < 3π/8 → LAD = |π/4 − LPA|
                        np.abs(3 * np.pi / 4 - lpa)  # 5π/8 < LPA < 7π/8 → LAD = |3π/4 − LPA|
                    )
                )
            )
        )
        
        # RAD calculation from RPA (Eq. 6)
        rad = np.where(
            (rpa >= 0) & (rpa <= pi_8), rpa,  # 0 ≤ RPA ≤ π/8 → RAD = RPA
            np.where(
                (rpa >= pi_78) & (rpa <= np.pi), np.pi - rpa,  # 7π/8 ≤ RPA ≤ π → RAD = π − RPA
                np.where(
                    (rpa >= pi_38) & (rpa <= pi_58), np.abs(np.pi / 2 - rpa),  # 3π/8 ≤ RPA ≤ 5π/8 → RAD = |π/2 − RPA|
                    np.where(
                        (rpa > pi_8) & (rpa < pi_38), np.abs(np.pi / 4 - rpa),  # π/8 < RPA < 3π/8 → RAD = |π/4 − RPA|
                        np.abs(3 * np.pi / 4 - rpa)  # 5π/8 < RPA < 7π/8 → RAD = |3π/4 − RPA|
                    )
                )
            )
        )
        
        # Step 5: Update PTI according to Algorithm 1 (Eq. 7)
        # PTI_i = LPT_i if LAD_i < RAD_i else RPT_i
        # In case of equality, choose RPT (else branch)
        self.pti = np.where(lad < rad, lpt, rpt)

        # Create new agents efficiently
        pop_new = []
        for idx in range(self.pop_size):
            pos_corrected = self.correct_solution(pos_new[idx])
            agent = self.generate_empty_agent(pos_corrected)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_corrected)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)

        # Perform greedy selection using standard Mealpy helper
        self.pop = pop_new
