#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:51, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import concurrent.futures as parallel
from functools import partial
import numpy as np
import time
from mealpy.optimizer import Optimizer


class BaseSRSR(Optimizer):
    """
        The original version of: Swarm Robotics Search And Rescue (SRSR)
            Swarm Robotics Search And Rescue: A Novel Artificial Intelligence-inspired Optimization Approach
        Link:
            https://doi.org/10.1016/j.asoc.2017.02.028
    """
    ID_POS = 0
    ID_FIT = 1
    ID_MU = 2
    ID_SIGMA = 3
    ID_POS_NEW = 4
    ID_FIT_NEW = 5
    ID_FIT_MOVE = 6

    def __init__(self, problem, epoch=10000, pop_size=100, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = 6 * pop_size
        self.epoch = epoch
        self.pop_size = pop_size

    def create_solution(self):
        """
        Returns:
            The position position with 2 element: index of position/location and index of fitness wrapper
            The general format: [position, [target, [obj1, obj2, ...]]]

        ## To get the position, fitness wrapper, target and obj list
        ##      A[self.ID_POS]                  --> Return: position
        ##      A[self.ID_FIT]                  --> Return: [target, [obj1, obj2, ...]]
        ##      A[self.ID_FIT][self.ID_TAR]     --> Return: target
        ##      A[self.ID_FIT][self.ID_OBJ]     --> Return: [obj1, obj2, ...]
        """
        position = np.random.uniform(self.problem.lb, self.problem.ub)
        fitness = self.get_fitness_position(position=position)
        mu = 0
        sigma = 0
        x_new = position.copy()
        fit_new = fitness.copy()
        fit_move = 0
        return [position, fitness, mu, sigma, x_new, fit_new, fit_move]

    def solve(self, mode='sequential'):
        """
        Args:
            mode (str): 'sequential', 'thread', 'process'
                + 'sequential': recommended for simple and small task (< 10 seconds for calculating objective)
                + 'thread': recommended for IO bound task, or small computing task (< 2 minutes for calculating objective)
                + 'process': recommended for hard and big task (> 2 minutes for calculating objective)

        Returns:
            [position, fitness value]
        """
        self.termination_start()
        pop = self.create_population(mode, self.pop_size)
        pop, g_best = self.get_global_best_solution(pop)  # We sort the population
        self.history.save_initial_best(g_best)

        # Control Parameters Of Algorithm
        # ==============================================================================================
        #  [c1] movement_factor : Determines Movement Pace Of Robots During Exploration Policy
        #  [c2] sigma_factor    : Determines Level Of Divergence Of Slave Robots From Master Robot
        #  [c3] sigma_limit     : Limits Value Of Sigma Factor
        #  [c4] mu_factor       : Controls Mean Value For Master And Slave Robots
        #       Control Parameters C1, C2 And C3 Are Automatically Tuned While C4 Should Be Set By User
        # ==============================================================================================
        mu_factor = 2 / 3  # [0.1-0.9] Controls Dominance Of Master Robot, Preferably 2/3
        sigma_temp = np.zeros(self.pop_size)  # Initializing Temporary Stacks
        SIF = None
        movement_factor = self.problem.ub - self.problem.lb

        for epoch in range(0, self.epoch):
            time_epoch = time.time()

            # ========================================================================================= %%
            #            PHASE 1 (ACCUMULATION): CALCULATING Mu AND SIGMA values FOR SOLUTIONS            %
            # ===========================================================================================%%

            # ------ CALCULATING MU AND SIGMA FOR MASTER ROBOT ----------
            pop[0][self.ID_SIGMA] = np.random.uniform()
            if epoch % 2 == 1:
                pop[0][self.ID_MU] = (1 - pop[0][self.ID_SIGMA]) * pop[0][self.ID_POS]
            else:
                pop[0][self.ID_MU] = (1 + (1 - mu_factor) * pop[0][self.ID_SIGMA]) * pop[0][self.ID_POS]

            for i in range(1, self.pop_size):
                # ---------- CALCULATING MU AND SIGMA FOR SLAVE ROBOTS ---------
                pop[i][self.ID_MU] = mu_factor * pop[0][self.ID_POS] + (1 - mu_factor) * pop[i][self.ID_POS]
                if epoch == 0:
                    SIF = 6
                sigma_temp[i] = SIF * np.random.uniform()
                pop[i][self.ID_SIGMA] = sigma_temp[i] * abs(pop[0][self.ID_POS] - pop[i][self.ID_POS]) + \
                                        np.random.uniform() ** 2 * ((pop[0][self.ID_POS] - pop[i][self.ID_POS]) < 0.05)

                # ----- Generating New Positions Using New Obtained Mu And Sigma Values --------------
                temp = np.random.normal(pop[i][self.ID_MU], pop[i][self.ID_SIGMA], self.problem.n_dims)
                pos_new = np.clip(temp, self.problem.lb, self.problem.ub)
                fit_new = self.get_fitness_position(pos_new)
                pop[i][self.ID_POS_NEW] = pos_new
                pop[i][self.ID_FIT_NEW] = fit_new

                # --------- Calculate Degree Of Cost Movement Of Robots During Movement --------------
                pop[i][self.ID_FIT_MOVE] = pop[i][self.ID_FIT][self.ID_TAR] - pop[i][self.ID_FIT_NEW][self.ID_TAR]

                # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------

                # Replace Solution If It Reached To A More Quality Position
                respec_id = int(np.floor(self.pop_size / (1 + round(np.random.uniform())))) - 1
                if self.compare_agent([pos_new, fit_new], pop[respec_id]):
                    pop[i][self.ID_POS] = pos_new.copy()
                    pop[i][self.ID_FIT] = fit_new.copy()
                else:
                    # Replace Solution Whether It Reached A Better Position Or Not
                    if self.compare_agent([pos_new, fit_new], pop[i]):
                        pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW].copy()
                        pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW].copy()

            # --------- Determining Sigma Improvement Factor (Sif) Based On Vvss Movement -------------------
            ## Get best improved fitness
            fit_id = np.argmax([item[self.ID_FIT_MOVE] for item in pop])
            sigma_factor = 1 + np.random.uniform() * np.max(self.problem.ub - self.problem.lb)
            SIF = sigma_factor * sigma_temp[fit_id]
            # Controlling Parameter Of Algorithm
            if SIF > np.max(self.problem.ub):
                SIF = np.max(self.problem.ub) * np.random.uniform()

            # ========================================================================================= %%
            #            Phase 2 (Exploration): Moving Slave Robots Toward Master Robot                   %
            # ===========================================================================================%%
            for i in range(0, self.pop_size):
                gb = np.random.uniform(-1, 1, self.problem.n_dims)
                gb[gb >= 0] = 1
                gb[gb < 0] = -1
                temp = pop[i][self.ID_POS] * np.random.uniform() + gb * (pop[0][self.ID_POS] - pop[i][self.ID_POS]) + \
                       movement_factor * np.random.uniform(self.problem.lb, self.problem.ub)
                pos_new = np.clip(temp, self.problem.lb, self.problem.ub)
                fit_new = self.get_fitness_position(pos_new)

                # ---------- Progress Assessment: Replacing More Quality Solutions With Previous Ones ------

                # Replace Solution If It Reached To A More Quality Position
                respec_id = int(np.floor(self.pop_size / (1 + round(np.random.uniform())))) - 1
                if self.compare_agent([pos_new, fit_new], pop[respec_id]):
                    pop[i][self.ID_POS] = pos_new.copy()
                    pop[i][self.ID_FIT] = fit_new.copy()
                else:
                    # Replace Solution Whether It Reached A Better Position Or Not
                    if self.compare_agent([pos_new, fit_new], pop[i]):
                        pop[i][self.ID_POS] = pop[i][self.ID_POS_NEW].copy()
                        pop[i][self.ID_FIT] = pop[i][self.ID_FIT_NEW].copy()

            # ========================================================================================= %%
            #        PHASE 3 (LOCAL SEARCH): CREATING SOME WORKER ROBOTS ASSIGNED TO SEARCH               %
            #                      LOCATIONS AROUND POSITION OF MASTER ROBOT                              %
            # ===========================================================================================%%

            if epoch > 0:
                # --- EXTRACTING "INTEGER PART" AND "FRACTIONAL PART"  OF THE ELEMENTS OF MASTER RPBOT POSITION------
                master_robot = {"original": np.reshape(pop[0][self.ID_POS], (self.problem.n_dims, 1)).copy(),
                                "sign": np.reshape(np.sign(pop[0][self.ID_POS]), (self.problem.n_dims, 1)).copy(),
                                "abs": np.reshape(abs(pop[0][self.ID_POS]), (self.problem.n_dims, 1)).copy(),
                                "int": np.reshape(np.floor(abs(pop[0][self.ID_POS])), (self.problem.n_dims, 1)).copy(),  # INTEGER PART
                                "frac": np.reshape(abs(pop[0][self.ID_POS]) - np.floor(abs(pop[0][self.ID_POS])), (self.problem.n_dims, 1)).copy()
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

                # --------- Proogress Assessment: Replacing More Quality Solutions With Previous Ones ---------------
                workers = np.concatenate((worker_robot1.T, worker_robot2.T, worker_robot3.T, worker_robot4.T, worker_robot5.T), axis=0)
                for i in range(0, 5):
                    workers[i] = np.clip(workers[i], self.problem.lb, self.problem.ub)
                    fit = self.get_fitness_position(workers[i])
                    if fit < pop[1][self.ID_FIT]:
                        pop[-(i + 1)][self.ID_POS] = workers[i].copy()
                        pop[-(i + 1)][self.ID_FIT] = fit.copy()

            pop, g_best = self.update_global_best_solution(pop)

            ## Additional information for the framework
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(pop.copy())
            self.print_epoch(epoch + 1, time_epoch)
            if self.termination_flag:
                if self.termination.mode == 'TB':
                    if time.time() - self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'FE':
                    self.count_terminate += self.nfe_per_epoch
                    if self.count_terminate >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                elif self.termination.mode == 'MG':
                    if epoch >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break
                else:  # Early Stopping
                    temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_FIT, self.ID_TAR, self.EPSILON)
                    if temp >= self.termination.quantity:
                        self.termination.logging(self.verbose)
                        break

        ## Additional information for the framework
        self.save_optimization_process()
        return self.solution[self.ID_POS], self.solution[self.ID_FIT][self.ID_TAR]
