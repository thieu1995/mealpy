# !/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy
from mealpy.optimizer import Optimizer


class OriginalSRSR(Optimizer):
    """
    The original version of: Swarm Robotics Search And Rescue (SRSR)

    Links:
        1. https://doi.org/10.1016/j.asoc.2017.02.028

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.swarm_based.SRSR import OriginalSRSR
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> model = OriginalSRSR(epoch, pop_size)
    >>> best_position, best_fitness = model.solve(problem_dict1)
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Bakhshipour, M., Ghadi, M.J. and Namdari, F., 2017. Swarm robotics search & rescue: A novel
    artificial intelligence-inspired optimization approach. Applied Soft Computing, 57, pp.708-726.
    """

    ID_POS = 0
    ID_TAR = 1
    ID_MU = 2
    ID_SIGMA = 3
    ID_POS_NEW = 4
    ID_FIT_NEW = 5
    ID_FIT_MOVE = 6

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = True

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        Overriding method in Optimizer class

        Returns:
            list: wrapper of solution with format [position, target, mu, sigma, x_new, target_new, target_move]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        mu = 0
        sigma = 0
        x_new = deepcopy(position)
        target_new = deepcopy(target)
        target_move = 0
        return [position, target, mu, sigma, x_new, target_new, target_move]

    def initialize_variables(self):
        # Control Parameters Of Algorithm
        # ==============================================================================================
        #  [c1] movement_factor : Determines Movement Pace Of Robots During Exploration Policy
        #  [c2] sigma_factor    : Determines Level Of Divergence Of Slave Robots From Master Robot
        #  [c3] sigma_limit     : Limits Value Of Sigma Factor
        #  [c4] mu_factor       : Controls Mean Value For Master And Slave Robots
        #       Control Parameters C1, C2 And C3 Are Automatically Tuned While C4 Should Be Set By User
        # ==============================================================================================
        self.mu_factor = 2 / 3  # [0.1-0.9] Controls Dominance Of Master Robot, Preferably 2/3
        self.sigma_temp = np.zeros(self.pop_size)  # Initializing Temporary Stacks
        self.SIF = None
        self.movement_factor = self.problem.ub - self.problem.lb

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # ========================================================================================= %%
        #            PHASE 1 (ACCUMULATION): CALCULATING Mu AND SIGMA values FOR SOLUTIONS            %
        # ===========================================================================================%%
        nfe_epoch = 0
        # ------ CALCULATING MU AND SIGMA FOR MASTER ROBOT ----------
        self.pop[0][self.ID_SIGMA] = np.random.uniform()
        if epoch % 2 == 1:
            self.pop[0][self.ID_MU] = (1 - self.pop[0][self.ID_SIGMA]) * self.pop[0][self.ID_POS]
        else:
            self.pop[0][self.ID_MU] = (1 + (1 - self.mu_factor) * self.pop[0][self.ID_SIGMA]) * self.pop[0][self.ID_POS]

        pop_new = []
        for i in range(0, self.pop_size):
            agent = deepcopy(self.pop[i])
            # ---------- CALCULATING MU AND SIGMA FOR SLAVE ROBOTS ---------
            self.pop[i][self.ID_MU] = self.mu_factor * self.pop[0][self.ID_POS] + (1 - self.mu_factor) * self.pop[i][self.ID_POS]
            if epoch == 0:
                self.SIF = 6
            self.sigma_temp[i] = self.SIF * np.random.uniform()
            self.pop[i][self.ID_SIGMA] = self.sigma_temp[i] * np.abs(self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS]) + \
                                         np.random.uniform() ** 2 * ((self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS]) < 0.05)

            # ----- Generating New Positions Using New Obtained Mu And Sigma Values --------------
            pos_new = np.random.normal(self.pop[i][self.ID_MU], self.pop[i][self.ID_SIGMA], self.problem.n_dims)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        nfe_epoch += self.pop_size

        for idx in range(0, self.pop_size):
            # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
            self.pop[idx][self.ID_FIT_MOVE] = self.pop[idx][self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_FIT_NEW][self.ID_FIT]

            self.pop[idx][self.ID_POS_NEW] = deepcopy(pop_new[idx][self.ID_POS])
            self.pop[idx][self.ID_FIT_NEW] = deepcopy(pop_new[idx][self.ID_TAR])

            # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------
            # Replace Solution If It Reached To A More Quality Position
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx][self.ID_POS] = deepcopy(pop_new[idx][self.ID_POS])
                self.pop[idx][self.ID_TAR] = deepcopy(pop_new[idx][self.ID_TAR])

        # --------- Determining Sigma Improvement Factor (Sif) Based On Vvss Movement -------------------
        ## Get best improved fitness
        fit_id = np.argmax([item[self.ID_FIT_MOVE] for item in self.pop])
        sigma_factor = 1 + np.random.uniform() * np.max(self.problem.ub - self.problem.lb)
        self.SIF = sigma_factor * self.sigma_temp[fit_id]
        # Controlling Parameter Of Algorithm
        if self.SIF > np.max(self.problem.ub):
            self.SIF = np.max(self.problem.ub) * np.random.uniform()

        # ========================================================================================= %%
        #            Phase 2 (Exploration): Moving Slave Robots Toward Master Robot                   %
        # ===========================================================================================%%
        pop_new = []
        for i in range(0, self.pop_size):
            agent = deepcopy(self.pop[i])
            gb = np.random.uniform(-1, 1, self.problem.n_dims)
            gb[gb >= 0] = 1
            gb[gb < 0] = -1
            pos_new = self.pop[i][self.ID_POS] * np.random.uniform() + gb * (self.pop[0][self.ID_POS] - self.pop[i][self.ID_POS]) + \
                   self.movement_factor * np.random.uniform(self.problem.lb, self.problem.ub)
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            agent[self.ID_POS] = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
        pop_new = self.update_target_wrapper_population(pop_new)
        nfe_epoch += self.pop_size

        for idx in range(0, self.pop_size):
            # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
            self.pop[idx][self.ID_FIT_MOVE] = self.pop[idx][self.ID_TAR][self.ID_FIT] - self.pop[idx][self.ID_FIT_NEW][self.ID_FIT]

            self.pop[idx][self.ID_POS_NEW] = deepcopy(pop_new[idx][self.ID_POS])
            self.pop[idx][self.ID_FIT_NEW] = deepcopy(pop_new[idx][self.ID_TAR])

            # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------
            # Replace Solution If It Reached To A More Quality Position
            if self.compare_agent(pop_new[idx], self.pop[idx]):
                self.pop[idx][self.ID_POS] = deepcopy(pop_new[idx][self.ID_POS])
                self.pop[idx][self.ID_TAR] = deepcopy(pop_new[idx][self.ID_TAR])

        # ========================================================================================= %%
        #        PHASE 3 (LOCAL SEARCH): CREATING SOME WORKER ROBOTS ASSIGNED TO SEARCH               %
        #                      LOCATIONS AROUND POSITION OF MASTER ROBOT                              %
        # ===========================================================================================%%

        if epoch > 0:
            # --- EXTRACTING "INTEGER PART" AND "FRACTIONAL PART"  OF THE ELEMENTS OF MASTER RPBOT POSITION------
            master_robot = {"original": deepcopy(np.reshape(self.pop[0][self.ID_POS], (self.problem.n_dims, 1))),
                            "sign": deepcopy(np.reshape(np.sign(self.pop[0][self.ID_POS]), (self.problem.n_dims, 1))),
                            "abs": deepcopy(np.reshape(abs(self.pop[0][self.ID_POS]), (self.problem.n_dims, 1))),
                            "int": deepcopy(np.reshape(np.floor(abs(self.pop[0][self.ID_POS])), (self.problem.n_dims, 1))),  # INTEGER PART
                            "frac": deepcopy(np.reshape(abs(self.pop[0][self.ID_POS]) - np.floor(abs(self.pop[0][self.ID_POS])), (self.problem.n_dims, 1)))
                            }  # FRACTIONAL PART

            # ------- Applying Nth-root And Nth-exponent Operators To Create Position Of New Worker Robots -------
            worker_robot1 = (master_robot["int"] + np.power(master_robot["frac"], 1 / (1 + np.random.randint(1, 4)))) * master_robot["sign"]
            id_changed1 = np.argwhere(np.round(np.random.uniform(self.problem.lb, self.problem.ub)))
            id_changed1 = np.reshape(id_changed1, (len(id_changed1)))
            worker_robot1 = np.reshape(worker_robot1, (self.problem.n_dims, 1))
            worker_robot1[id_changed1] = master_robot["original"][id_changed1]

            worker_robot2 = (master_robot["int"] + np.power(master_robot["frac"], (1 + np.random.randint(1, 4)))) * master_robot["sign"]
            id_changed2 = np.argwhere(np.round(np.random.uniform(self.problem.lb, self.problem.ub)))
            id_changed2 = np.reshape(id_changed2, (len(id_changed2)))
            worker_robot2 = np.reshape(worker_robot2, (self.problem.n_dims, 1))
            worker_robot2[id_changed2] = master_robot["original"][id_changed2]

            # -------- Applying A Combined Ga-like Operator To Create Position Of New Worker Robot -------------
            random_per_mutation = np.random.permutation(self.problem.n_dims)
            sec1 = random_per_mutation[0: int(self.problem.n_dims / 2)]
            sec2 = random_per_mutation[int(self.problem.n_dims / 2):]
            worker_robot3 = np.zeros((self.problem.n_dims, 1))
            worker_robot3[sec1] = (master_robot["int"][sec1] + np.power(master_robot["frac"][sec1],
                                                                        1 / (1 + np.random.randint(1, 4)))) * master_robot["sign"][sec1]
            worker_robot3[sec2] = (master_robot["int"][sec2] + master_robot["frac"][sec2] **
                                   (1 + np.random.randint(1, 4))) * master_robot["sign"][sec2]
            id_changed3 = np.argwhere(np.round(np.random.uniform(self.problem.lb, self.problem.ub)))
            id_changed3 = np.reshape(id_changed3, (len(id_changed3)))
            worker_robot3[id_changed3] = master_robot["original"][id_changed3]

            # ------- Applying Round Operators To Create Position Of New Worker Robot -------------------
            worker_robot4 = np.ceil(master_robot["abs"]) * master_robot["sign"]
            id_changed4 = np.argwhere(np.round(np.random.uniform(self.problem.lb, self.problem.ub)))
            id_changed4 = np.reshape(id_changed4, (len(id_changed4)))
            worker_robot4[id_changed4] = master_robot["original"][id_changed4]

            worker_robot5 = np.floor(master_robot["abs"]) * master_robot["sign"]
            id_changed5 = np.argwhere(np.round(np.random.uniform(self.problem.lb, self.problem.ub)))
            id_changed5 = np.reshape(id_changed5, (len(id_changed5)))
            worker_robot5[id_changed5] = master_robot["original"][id_changed5]

            # --------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ---------------
            workers = np.concatenate((worker_robot1.T, worker_robot2.T, worker_robot3.T, worker_robot4.T, worker_robot5.T), axis=0)
            pop_workers = []
            for i in range(0, 5):
                pos_new = self.amend_position(workers[i], self.problem.lb, self.problem.ub)
                pop_workers.append([pos_new, None])
                if self.mode not in self.AVAILABLE_MODES:
                    pop_workers[-1][self.ID_TAR] = self.get_target_wrapper(pos_new)
            pop_workers = self.update_target_wrapper_population(pop_workers)
            nfe_epoch += 5

            for i in range(0, 5):
                if self.compare_agent(pop_workers[i], self.pop[1]):
                    self.pop[-(i + 1)][self.ID_POS] = deepcopy(pop_workers[i][self.ID_POS])
                    self.pop[-(i + 1)][self.ID_TAR] = deepcopy(pop_workers[i][self.ID_TAR])
            self.nfe_per_epoch = nfe_epoch
