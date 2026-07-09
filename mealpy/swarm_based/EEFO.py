# This code is ported from the original MATLAB implementation:
# Copyright (c) 2023, W. Zhao (BSD 3-Clause License)
import numpy as np
import math
from mealpy.optimizer import Optimizer


class EEFO(Optimizer):
    """
    Electric Eel Foraging Optimization (EEFO)

    Links:
        1. https://doi.org/10.1016/j.eswa.2023.122200
        2. https://www.mathworks.com/matlabcentral/fileexchange/153461-electric-eel-foraging-optimization-eefo

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + epoch (int): Maximum number of iterations, default = 10000
        + pop_size (int): Population size, default = 100

    Notes:
        1. The code is adapted 1:1 from the original MATLAB implementation by W. Zhao et al. (2023).
        2. Implements specific boundary handling (random re-initialization for violated dims) as per 'SpaceBound.m'.
        3. Uses standard normal distribution for Levy flight step calculations.

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar
    >>> from mealpy.swarm_based.EEFO import EEFO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=[-100.] * 30, ub=[100.] * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EEFO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L., Zhang, Z., Fan, H., Zhang, J., Mirjalili, S., ... & Cao, Q. (2024).
    Electric eel foraging optimization: A new bio-inspired optimizer for engineering applications.
    Expert Systems with Applications, 238, 122200.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): Maximum number of iterations, default = 10000
            pop_size (int): Population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def _levy(self, dim):
        """
        Levy flight implementation based on the original MATLAB 'levy.m'.
        Uses Gamma function and standard normal distribution.
        """
        beta = 1.5
        # MATLAB: s=(gamma(1+b)*sin(pi*b/2)/(gamma((1+b)/2)*b*2^((b-1)/2)))^(1/b);
        num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma = (num / den) ** (1 / beta)

        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)

        # MATLAB: sigma=u./abs(v).^(1/b);
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _space_bound(self, pos, lb, ub):
        """
        Boundary handling based on the original MATLAB 'SpaceBound.m'.
        Unlike standard clipping, this method re-initializes ONLY the dimensions
        that violate bounds randomly within [lb, ub].
        """
        # Create a boolean mask where True indicates out of bounds
        is_out = (pos > ub) | (pos < lb)

        # If any dimension is out of bounds
        if np.any(is_out):
            # Generate random values for the whole vector (to pick from)
            random_pos = np.random.uniform(lb, ub)
            # Replace only the out-of-bound dimensions with random values
            pos = np.where(is_out, random_pos, pos)
        return pos

    def evolve(self, epoch):
        """
        The main evolution process of EEFO algorithm.
        """
        it = epoch
        max_it = self.epoch
        dim = self.problem.n_dims
        lb = self.problem.lb
        ub = self.problem.ub

        # Population mean position (required for Eqs. 20, 24, 25)
        pop_pos_matrix = np.array([agent.solution for agent in self.pop])
        mean_pop_pos = np.mean(pop_pos_matrix, axis=0)

        pop_new = []

        # Eq. (30): Energy factor E0 calculation
        e0 = 4 * np.sin(1 - it / max_it)

        for idx in range(0, self.pop_size):
            agent = self.pop[idx]
            x = agent.solution

            # Eq. (30): Energy factor E calculation
            # MATLAB: E=E0*log(1/rand);
            e_factor = e0 * np.log(1 / np.random.rand())

            # --- Direct Vector Calculation ---
            # Used for determining which dimensions to update in the interaction phase
            direct_vector = np.zeros(dim)
            if dim == 1:
                direct_vector[:] = 1
            else:
                # MATLAB: RandNum=ceil((MaxIt-It)/MaxIt*rand*(Dim-2)+2);
                rand_val = np.random.rand()
                rand_num = int(np.ceil((max_it - it) / max_it * rand_val * (dim - 2) + 2))

                # MATLAB: RandDim=randperm(Dim);
                rand_dim = np.random.permutation(dim)

                # MATLAB: DirectVector(i,RandDim(1:RandNum))=1;
                direct_vector[rand_dim[:rand_num]] = 1

            pos_new = x.copy()

            # --- PHASE 1: Exploration (Interaction) ---
            # Active when Energy Factor > 1
            if e_factor > 1:
                # Select a random partner 'j' distinct from current agent 'i'
                candidates = list(range(0, idx)) + list(range(idx + 1, self.pop_size))
                j = np.random.choice(candidates)
                agent_j = self.pop[j]

                # Eq. (7): Interaction based on fitness comparison
                if self.compare_target(agent_j.target, agent.target):
                    if np.random.rand() > 0.5:
                        pos_new = agent_j.solution + np.random.normal() * direct_vector * (mean_pop_pos - x)
                    else:
                        xr = np.random.uniform(lb, ub)
                        pos_new = agent_j.solution + 1 * np.random.normal() * direct_vector * (xr - x)
                else:
                    if np.random.rand() > 0.5:
                        pos_new = x + np.random.normal() * direct_vector * (mean_pop_pos - agent_j.solution)
                    else:
                        xr = np.random.uniform(lb, ub)
                        pos_new = x + np.random.normal() * direct_vector * (xr - agent_j.solution)

            # --- PHASE 2: Exploitation ---
            # Active when Energy Factor <= 1
            else:
                rand_prob = np.random.rand()

                # Mode A: Resting (Eq. 16)
                if rand_prob < 1/3:
                    # Eq. (15): Alpha calculation
                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())

                    rn = np.random.randint(0, self.pop_size)
                    rd = np.random.randint(0, dim)
                    agent_rn = self.pop[rn]

                    # Eq. (12) & (13): Z vector calculation
                    z_scalar = (agent_rn.solution[rd] - lb[rd]) / (ub[rd] - lb[rd])
                    z_vec = lb + z_scalar * (ub - lb)

                    # Eq. (14): Ri calculation (Interaction with global best)
                    r_i = z_vec + alpha * np.abs(z_vec - self.g_best.solution)

                    # Eq. (16): Position update
                    pos_new = r_i + np.random.normal() * (r_i - np.round(np.random.rand()) * x)

                # Mode B: Migrating (Eq. 24)
                elif rand_prob > 2/3:
                    rn = np.random.randint(0, self.pop_size)
                    rd = np.random.randint(0, dim)
                    agent_rn = self.pop[rn]

                    z_scalar = (agent_rn.solution[rd] - lb[rd]) / (ub[rd] - lb[rd])
                    z_vec = lb + z_scalar * (ub - lb)

                    alpha = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())
                    r_i = z_vec + alpha * np.abs(z_vec - self.g_best.solution)

                    # Eq. (21): Beta calculation
                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())

                    # Eq. (25): Hr calculation (Hunting area)
                    hr = self.g_best.solution + beta * np.abs(mean_pop_pos - self.g_best.solution)

                    # Eq. (26): Levy flight
                    l_vec = 0.01 * np.abs(self._levy(dim))

                    # Eq. (24): Position update
                    pos_new = -np.random.rand() * r_i + np.random.rand() * hr - l_vec * (hr - x)

                # Mode C: Hunting (Eq. 22)
                else:
                    # Eq. (21): Beta calculation
                    beta = 2 * (np.exp(1) - np.exp(it / max_it)) * np.sin(2 * np.pi * np.random.rand())

                    # Eq. (20): Hprey calculation
                    h_prey = self.g_best.solution + beta * np.abs(mean_pop_pos - self.g_best.solution)

                    r4 = np.random.rand()
                    # Eq. (23): Eta calculation
                    eta = np.exp(r4 * (1 - it) / max_it) * np.cos(2 * np.pi * r4)

                    # Eq. (22): Position update
                    pos_new = h_prey + eta * (h_prey - np.round(np.random.rand()) * x)

            # --- Boundary Handling ---
            # Using specific boundary handling from SpaceBound.m
            pos_new = self._space_bound(pos_new, lb, ub)

            # --- Create New Agent ---
            # Fitness is calculated implicitly in generate_agent
            agent_new = self.generate_agent(pos_new)
            pop_new.append(agent_new)

        # --- Greedy Selection Mechanism ---
        # Update population only if the new position provides better fitness
        for idx in range(self.pop_size):
            if self.compare_target(pop_new[idx].target, self.pop[idx].target):
                self.pop[idx] = pop_new[idx]