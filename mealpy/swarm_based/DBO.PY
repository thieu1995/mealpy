import numpy as np
from mealpy.optimizer import Optimizer

class DBO(Optimizer):
    """
    Dung Beetle Optimizer (DBO)
    
    Links:
        1. https://doi.org/10.1007/s11227-022-04959-6
        
    Notes:
        The algorithm mimics the ball-rolling, dancing, foraging, stealing, and reproduction behaviors of dung beetles.
        Implementation based on the original paper logic.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (evolve) of the algorithm.
        """
        # Define population size for each group based on paper (approximate percentages)
        n_roll = int(self.pop_size * 0.2)
        n_brood = int(self.pop_size * 0.2)
        n_small = int(self.pop_size * 0.25)
        # The rest are thieves
        
        # Current Global Best and Worst
        pos_best = self.g_best[self.ID_POS]           # X^b
        pos_worst = self.pop[-1][self.ID_POS]         # X^w
        pos_local_best = self.pop[0][self.ID_POS]     # X^*

        # R factor (decreases linearly)
        R = 1 - (epoch / self.epoch)

        pop_new = []
        for idx in range(0, self.pop_size):
            pos_now = self.pop[idx][self.ID_POS]
            pos_new = pos_now.copy()

            # --- 1. Ball-Rolling Dung Beetles ---
            if idx < n_roll:
                # Obstacle simulation (prob = 0.9)
                if np.random.rand() < 0.9:
                    # Rolling with celestial cues (Eq. 1)
                    alpha = 1 if np.random.rand() > 0.5 else -1
                    k = 0.1
                    b = 0.3
                    delta_x = np.abs(pos_now - pos_worst)
                    pos_new = pos_now + alpha * k * pos_now + b * delta_x
                else:
                    # Dancing (Eq. 2)
                    theta = np.random.uniform(0, np.pi)
                    if theta != 0 and theta != np.pi / 2 and theta != np.pi:
                        pos_new = pos_now + np.tan(theta) * np.abs(pos_now)

            # --- 2. Brood Balls (Reproduction) ---
            elif idx < n_roll + n_brood:
                # Spawning area boundaries (Eq. 3)
                lb_star = np.maximum(pos_local_best * (1 - R), self.problem.lb)
                ub_star = np.minimum(pos_local_best * (1 + R), self.problem.ub)
                
                # Position update (Eq. 4)
                b1 = np.random.rand(self.problem.n_dims)
                b2 = np.random.rand(self.problem.n_dims)
                pos_new = pos_local_best + b1 * (pos_now - lb_star) + b2 * (pos_now - ub_star)

            # --- 3. Small Dung Beetles (Foraging) ---
            elif idx < n_roll + n_brood + n_small:
                # Foraging area (Eq. 5)
                lb_b = np.maximum(pos_best * (1 - R), self.problem.lb)
                ub_b = np.minimum(pos_best * (1 + R), self.problem.ub)
                
                # Position update (Eq. 6)
                C1 = np.random.normal(0, 1)
                C2 = np.random.rand(self.problem.n_dims)
                pos_new = pos_now + C1 * (pos_now - lb_b) + C2 * (pos_now - ub_b)

            # --- 4. Thieves (Stealing) ---
            else:
                # Stealing behavior (Eq. 7)
                S = 0.5
                g = np.random.normal(0, 1, self.problem.n_dims)
                pos_new = pos_best + S * g * (np.abs(pos_now - pos_local_best) + np.abs(pos_now - pos_best))

            # Boundary check
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append([pos_new, None])
            
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)

        self.pop = self.update_target_wrapper_population(pop_new)