#!/usr/bin/env python
# Implemented for MEALPY by Furkan Gunbaz (gunbaz)
# This implementation reproduces the algorithm exactly as described in 
# mathematics-13-01500-v2.pdf (MShOA), with no simplifications.
# Email: furkan.gunbaz@gmail.com
# Github: https://github.com/gunbaz
# -------------------------------------------------
#
# IMPLEMENTATION NOTES:
# This implementation follows the PTI (Polarization Type Indicator) mechanism 
# from the original paper exactly as described in Algorithm 1 and Algorithm 2.
# Key features:
# 1. PTI Update (Algorithm 1): LPA, RPA, LPT, RPT, LAD, RAD calculations (Eq. 5, 6, 7)
# 2. Strategy 1 - Foraging: Langevin/Brownian equation (Eq. 12)
# 3. Strategy 2 - Attack/Strike: Circular motion equation (Eq. 14)
# 4. Strategy 3 - Defense/Burrow: Defense/Shelter split with k parameter (Eq. 15)
#
# CRITICAL: LPA is calculated from intra-iteration change (X_i(t) vs X'_i(t)),
# not inter-iteration change. PTI update happens AFTER strategy application.

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalMShOA(Optimizer):
    """
    The original version of: Mantis Shrimp Optimization Algorithm (MShOA)
    
    This implementation reproduces the algorithm exactly as described in 
    mathematics-13-01500-v2.pdf, with no simplifications.
    
    Each agent has a PTI (Polarization Type Indicator) value ∈ {1, 2, 3} that determines strategy:
    - PTI = 1: Foraging/Navigation (vertical linear polarized light detection) → Strategy 1
    - PTI = 2: Attack/Strike (horizontal linear polarized light detection) → Strategy 2
    - PTI = 3: Defense/Burrow (circular polarized light detection) → Strategy 3
    
    The PTI vector is initialized randomly and updated each iteration according to Algorithm 1.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + polarization_rate (float): [DEPRECATED] Kept for backward compatibility, not used
        + strike_factor (float): [DEPRECATED] Not used in original equation (Equation 14 uses circular motion)
        + k_value (float): [0.0, 1.0], upper bound for k parameter in defense/shelter phase (Strategy 3, Eq. 15). 
                          k is sampled from U(0, k_value). Default = 0.3 (matches paper value).

    Links:
        1. https://doi.org/10.3390/math13091500

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
    >>> model = MShOA.OriginalMShOA(epoch=1000, pop_size=50, polarization_rate=0.5, strike_factor=1.5, k_value=0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sánchez Cortez, J.A., Peraza Vázquez, H., Peña Delgado, A.F., 2025. Mantis Shrimp Optimization Algorithm (MShOA): A Novel Bio-Inspired Optimization Algorithm Based on Mantis Shrimp Survival Tactics. Mathematics, 13(9), 1500. https://doi.org/10.3390/math13091500
    
    Notes
    ~~~~~
    This implementation uses PTI-based strategy selection exactly as described in Algorithm 1 and Algorithm 2.
    All equations match the paper exactly:
    - Algorithm 1: PTI update mechanism (Eq. 5, 6, 7)
    - Strategy 1: Foraging equation (Eq. 12)
    - Strategy 2: Attack/Strike equation (Eq. 14)
    - Strategy 3: Defense/Burrow equation (Eq. 15)
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, polarization_rate: float = 0.5,
                 strike_factor: float = 1.5, k_value: float = 0.3, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            polarization_rate: [DEPRECATED - kept for backward compatibility] 
                              PTI mechanism now handles strategy selection automatically
            strike_factor: [DEPRECATED] Not used in original equation (Equation 14 uses circular motion)
            k_value: Upper bound for k parameter in defense/shelter phase (Strategy 3, Equation 15). 
                    k is sampled from U(0, k_value). Default = 0.3 (matches paper value).
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        # Keep polarization_rate for backward compatibility but it's not used
        self.polarization_rate = self.validator.check_float("polarization_rate", polarization_rate, (0.0, 1.0))
        self.strike_factor = self.validator.check_float("strike_factor", strike_factor, (0.0, 5.0))
        self.k_value = self.validator.check_float("k_value", k_value, (0.0, 1.0))
        self.set_parameters(["epoch", "pop_size", "polarization_rate", "strike_factor", "k_value"])
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
        super().before_main_loop()
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
        
        # Use standard Mealpy helper to update all targets
        pop_new = self.update_target_for_population(pop_new)
        
        # Safety check: ensure no agent has None target
        for agent in pop_new:
            if agent.target is None:
                agent.target = self.get_target(agent.solution)
        
        # Perform greedy selection using standard Mealpy helper
        self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
