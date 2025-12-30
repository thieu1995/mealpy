#!/usr/bin/env python
# Created by "Antigravity" at 2025

import numpy as np
from mealpy.optimizer import Optimizer


class MLFA_GD(Optimizer):
    """
    MLFA-GD: Moderate Firefly Algorithm with Gender Difference (or Firefly Algorithm with Multiple Learning Ability based on Gender Difference)
    
    Links:
        1. https://doi.org/10.1038/s41598-025-09523-9
    
    Notes:
        Implementation based on the local PDF at D:\\Projects\\mealpy_v2\\MLFA-GD.pdf
        Scientific Reports, 2025 (Volume 15, Article 28400).
        
        The algorithm introduces:
        1. Population division into Male and Female subgroups.
        2. Male fireflies strategy: Partial Attraction Model with Escape Mechanism (Eq. 8).
        3. Female fireflies strategy: Dual Elites Guided Learning (Eq. 12, 13).
        4. General Centroid Deep Learning (Eq. 11).
        5. Global Best Random Walk (Eq. 14).
        
    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): maximum number of iterations, default = 1000
        + pop_size (int): number of population size, default = 50
        + gamma (float): Light Absorption Coefficient, default = 1.0
        + beta_base (float): Attraction Coefficient Base Value, default = 1.0
        + alpha (float): scaling parameter (legacy/unused in core new equations but kept for structure), default = 0.2
        + m_females (int): number of females to learn from (m), default = 3
        + learning_count (int): deep learning count for centroid (count), default = 250
        + k_walk (int): chaotic random walk steps (k), default = 5
    """
    
    def __init__(self, epoch: int = 1000, pop_size: int = 50, gamma: float = 1.0, 
                 beta_base: float = 1.0, alpha: float = 0.2, m_females: int = 3, 
                 learning_count: int = 250, k_walk: int = 5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 1000
            pop_size (int): number of population size, default = 50
            gamma (float): Light Absorption Coefficient, default = 1.0
            beta_base (float): Attraction Coefficient Base Value, default = 1.0
            alpha (float): scaling parameter, default = 0.2
            m_females (int): number of females to learn from, default = 3
            learning_count (int): deep learning count for centroid, default = 250
            k_walk (int): chaotic random walk steps, default = 5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.gamma = self.validator.check_float("gamma", gamma, (0, 10.0))
        self.beta_base = self.validator.check_float("beta_base", beta_base, (0, 10.0))
        self.alpha = self.validator.check_float("alpha", alpha, (0, 10.0))
        self.m_females = self.validator.check_int("m_females", m_females, [1, self.pop_size])
        self.learning_count = self.validator.check_int("learning_count", learning_count, [0, 10000])
        self.k_walk = self.validator.check_int("k_walk", k_walk, [1, 100])
        
        self.set_parameters(["epoch", "pop_size", "gamma", "beta_base", "alpha", "m_females", "learning_count", "k_walk"])
        self.sort_flag = False
        
    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Initialize male_pbests in the first epoch
        if epoch == 1:
            self.n_males = int(np.ceil(self.pop_size / 2))
            self.n_females = self.pop_size - self.n_males
            self.male_pbests = [agent.copy() for agent in self.pop[:self.n_males]]

        # Split population

        males = self.pop[:self.n_males]
        females = self.pop[self.n_males:]
        
        # ==========================
        # 1. Update Male Fireflies (Algorithm 1)
        # ==========================
        pop_new_males = []
        for i in range(self.n_males):
            agent = males[i]
            
            # Select m random females
            current_females_len = len(females)
            if current_females_len > 0:
                n_select = min(self.m_females, current_females_len)
                selected_indices = self.generator.choice(current_females_len, size=n_select, replace=False)
                selected_females = [females[idx] for idx in selected_indices]
            else:
                selected_females = []

            # Calculate movement accumulation (Eq. 8)
            movement_accum = np.zeros(self.problem.n_dims)
            
            for female in selected_females:
                # Compare fitness (target.fitness). minimize: smaller is better.
                # d_k = 1 if female is brighter (better), -1 else
                if self.compare_target(female.target, agent.target, self.problem.minmax): 
                    # female is better
                    d_k = 1.0
                else:
                    d_k = -1.0
                
                # Distance
                dist = np.linalg.norm(agent.solution - female.solution)
                beta = self.beta_base * np.exp(-self.gamma * (dist ** 2))
                
                # Lambda: random number from 0 to 1
                lam = self.generator.uniform(0, 1)
                
                movement_accum += d_k * beta * lam * (female.solution - agent.solution)
            
            # Eq. 8 (pdf)
            # alpha_i(t) * epsilon_i . Scale with (ub-lb) helps in large domains, but paper implies simple randomness.
            # Removing (ub-lb) as per verification findings for fine-tuning.
            movement_accum += self.alpha * self.generator.uniform(-0.5, 0.5, self.problem.n_dims)
            
            # Update position (single increment)
            pos_new = agent.solution + movement_accum
            pos_new = self.correct_solution(pos_new)
            new_agent = self.generate_empty_agent(pos_new)
            pop_new_males.append(new_agent)
            
        # Update fitness for new males
        pop_new_males = self.update_target_for_population(pop_new_males)
        # In single/process/thread mode, if logic differs, ensure fitness is calculated
        if pop_new_males[0].target is None:
            for agent in pop_new_males:
                agent.target = self.get_target(agent.solution)
        
        # Update Male PBests and Global Best
        for i in range(self.n_males):
            # Update PBest
            if self.compare_target(pop_new_males[i].target, self.male_pbests[i].target, self.problem.minmax):
                self.male_pbests[i] = pop_new_males[i].copy()
            
            # Update curr male population
            self.pop[i] = pop_new_males[i]
            
            # Update Global Best
            if self.compare_target(self.pop[i].target, self.g_best.target, self.problem.minmax):
                self.g_best = self.pop[i].copy()
        
        # Decay alpha to reduce randomness over time for fine-tuning
        self.alpha = self.alpha * 0.99
                
        # ==========================
        # 2. General Centroid and Deep Learning
        # ==========================
        # Eq 10: yGC calculated from male pbests
        pbest_solutions = np.array([agent.solution for agent in self.male_pbests])
        yGC = np.mean(pbest_solutions, axis=0) # Centroid position
        
        # Eq 11: Deep Learning for yGC (Algorithm 4 Step 14)
        for _ in range(self.learning_count):
            # Pick random male r
            r_idx = self.generator.integers(0, self.n_males)
            y_r = self.pop[r_idx].solution # y_r(t)
            
            cauchy_vec = self.generator.standard_cauchy(self.problem.n_dims)
            yGC = yGC + cauchy_vec * (y_r - yGC)
            
        # Evaluate yGC fitness (needed for Female update)
        yGC = self.correct_solution(yGC)
        yGC_agent = self.generate_empty_agent(yGC)
        yGC_agent.target = self.get_target(yGC_agent.solution)
        
        # ==========================
        # 3. Update Female Fireflies (Algorithm 2)
        # ==========================
        pop_new_females = []
        for i in range(self.n_females):
            agent = females[i]
            
            # Compare with yGC
            if self.compare_target(yGC_agent.target, agent.target, self.problem.minmax):
                # Eq 12: Move toward yGC
                dist = np.linalg.norm(agent.solution - yGC_agent.solution)
                beta = self.beta_base * np.exp(-self.gamma * (dist ** 2))
                pos_new = agent.solution + beta * (yGC_agent.solution - agent.solution)
            else:
                # Eq 13: Move toward xgbest with Cauchy mutation
                cauchy_vec = self.generator.standard_cauchy(self.problem.n_dims)
                pos_new = self.g_best.solution + cauchy_vec * self.alpha
                
            pos_new = self.correct_solution(pos_new)
            new_agent = self.generate_empty_agent(pos_new)
            pop_new_females.append(new_agent)
            
        # Update fitness for females
        pop_new_females = self.update_target_for_population(pop_new_females)
        if pop_new_females[0].target is None:
            for agent in pop_new_females:
                agent.target = self.get_target(agent.solution)
        
        # Update Females and Global Best
        for i in range(self.n_females):
            pop_idx = self.n_males + i
            self.pop[pop_idx] = pop_new_females[i]
            
            if self.compare_target(pop_new_females[i].target, self.g_best.target, self.problem.minmax):
                self.g_best = pop_new_females[i].copy()

        # ==========================
        # 4. Random Walk for Global Best (Algorithm 3)
        # ==========================
        # Eq 14: epsilon calculation
        # Use simple cubic decay for fine convergence
        epsilon = ((self.epoch - epoch + 1) / self.epoch) ** 3
        
        # Logistic map initialization (chaotic number)
        ch_val = 0.7 
        
        for _ in range(self.k_walk):
            # Generate D-dimensional chaotic vector (Clarification A)
            # Instead of broadcasting one scalar, we iterate the logistic map D times
            # to create a vector of chaotic values for this step.
            chaotic_vector = np.zeros(self.problem.n_dims)
            for d in range(self.problem.n_dims):
                ch_val = 4 * ch_val * (1 - ch_val)
                chaotic_vector[d] = ch_val
            
            # Map chaotic vector [0,1]^D to input space [lb, ub]^D
            # Eq 14: xgbest' = (1-epsilon) * xgbest + epsilon * mapped_val
            mapped_vec = self.problem.lb + chaotic_vector * (self.problem.ub - self.problem.lb)
            
            xgbest_prime_pos = (1 - epsilon) * self.g_best.solution + epsilon * mapped_vec
            xgbest_prime_pos = self.correct_solution(xgbest_prime_pos)
            
            # Evaluate
            xgbest_prime_agent = self.generate_empty_agent(xgbest_prime_pos)
            xgbest_prime_agent.target = self.get_target(xgbest_prime_pos)
            
            # Update if better
            if self.compare_target(xgbest_prime_agent.target, self.g_best.target, self.problem.minmax):
                self.g_best = xgbest_prime_agent.copy()
